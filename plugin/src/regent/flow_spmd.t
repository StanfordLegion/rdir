-- Copyright (c) 2015-2017, Stanford University. All rights reserved.
--
-- This file is dual-licensed under the BSD license (shown below) and
-- Apache version 2.0 license.
--
-- Redistribution and use in source and binary forms, with or without
-- modification, are permitted provided that the following conditions
-- are met:
--  * Redistributions of source code must retain the above copyright
--    notice, this list of conditions and the following disclaimer.
--  * Redistributions in binary form must reproduce the above copyright
--    notice, this list of conditions and the following disclaimer in the
--    documentation and/or other materials provided with the distribution.
--  * Neither the name of NVIDIA CORPORATION nor the names of its
--    contributors may be used to endorse or promote products derived
--    from this software without specific prior written permission.
--
-- THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
-- EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
-- IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
-- PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
-- CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
-- EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
-- PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
-- PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
-- OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
-- (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
-- OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

-- Dataflow-based SPMD Optimization

local ast = require("regent/ast")
local data = require("common/data")
local codegen = require("regent/codegen")
local flow = require("regent/flow")
local flow_extract_subgraph = require("regent/flow_extract_subgraph")
local flow_outline_task = require("regent/flow_outline_task")
local flow_region_tree = require("regent/flow_region_tree")
local flow_summarize_subgraph = require("regent/flow_summarize_subgraph")
local flow_to_ast = require("regent/flow_to_ast")
local optimize_config_options = require("regent/optimize_config_options")
local optimize_divergence = require("regent/optimize_divergence")
local optimize_futures = require("regent/optimize_futures")
local optimize_index_launches = require("regent/optimize_index_launches")
local optimize_mapping = require("regent/optimize_mapping")
local optimize_traces = require("regent/optimize_traces")
local passes_hooks = require("regent/passes_hooks")
local pretty = require("regent/pretty")
local report = require("common/report")
local std = require("regent/std")
local vectorize_loops = require("regent/vectorize_loops")

local spmd_mapping = require("regent/flow_mapping")

-- Configuration Variables

-- Setting this flag configures the number of SPMD tasks assigned to
-- each shard. Generally speaking, this should be approximately equal
-- to the number of tasks per node.
local shard_size = std.config["flow-spmd-shardsize"]

local context = {}

function context:__index (field)
  local value = context [field]
  if value ~= nil then
    return value
  end
  error ("context has no field '" .. field .. "' (in lookup)", 2)
end

function context:__newindex (field, value)
  error ("context has no field '" .. field .. "' (in assignment)", 2)
end

function context:new_graph_scope(graph)
  assert(flow.is_graph(graph))
  local cx = {
    tree = graph.region_tree,
    graph = graph,
  }
  return setmetatable(cx, context)
end

function context.new_global_scope()
  local cx = {}
  return setmetatable(cx, context)
end

local flow_spmd = {}

local function is_none(edge)
  return edge.label:is(flow.edge.None)
end
local function is_read(edge)
  return edge.label:is(flow.edge.Read)
end
local function is_write(edge)
  return edge.label:is(flow.edge.Write)
end
local function is_reduce(edge)
  return edge.label:is(flow.edge.Reduce)
end
local function is_read_write(edge)
  return edge.label:is(flow.edge.Read) or edge.label:is(flow.edge.Write)
end
local function is_write_reduce(edge)
  return edge.label:is(flow.edge.Write) or edge.label:is(flow.edge.Reduce)
end
local function is_any_privilege(edge)
  return edge.label:is(flow.edge.None) or edge.label:is(flow.edge.Read) or
    edge.label:is(flow.edge.Write) or edge.label:is(flow.edge.Reduce)
end
local function is_happens_before(edge)
  return edge.label:is(flow.edge.HappensBefore)
end

local function isnt_happens_before(edge)
  return not edge.label:is(flow.edge.HappensBefore)
end

local function is_leaf(cx, nid)
  local label = cx.graph:node_label(nid)
  if not label:is(flow.node.ctrl.ForNum) and
     not label:is(flow.node.ctrl.ForList) then return false end

  local block_cx = cx:new_graph_scope(label.block)
  local inner = block_cx.graph:any_nodes(
    function(_, label) return label:is(flow.node.ctrl) end)
  local task = block_cx.graph:any_nodes(
    function(_, label) return label:is(flow.node.Task) end)
  return task and not inner
end

local function is_inner(cx, nid)
  local label = cx.graph:node_label(nid)
  return label:is(flow.node.ctrl) and not is_leaf(cx, nid)
end

local function has_demand_spmd(cx, nid)
  local label = cx.graph:node_label(nid)
  return label.annotations.spmd:is(ast.annotation.Demand)
end

local function whitelist(label)
  return label:is(flow.node.Binary) or
    label:is(flow.node.Cast) or
    label:is(flow.node.IndexAccess) or
    label:is(flow.node.Assignment) or
    label:is(flow.node.Reduce) or
    (label:is(flow.node.Task) and not label.opaque) or
    label:is(flow.node.Open) or
    label:is(flow.node.Close) or
    label:is(flow.node.ctrl) or
    label:is(flow.node.data) or
    label:is(flow.node.Constant) or
    label:is(flow.node.Function)
end

local function whitelist_node_types(cx, loop_nid)
  local loop_label = cx.graph:node_label(loop_nid)
  local block_cx = cx:new_graph_scope(loop_label.block)
  local result = block_cx.graph:traverse_nodes_recursive(
    function(graph, nid, label)
      return not whitelist(label) and {graph, nid} or nil
    end)
  if result == nil then
    return true
  else
    return false, unpack(result)
  end
end

local function has_leaves(cx, loop_nid)
  local loop_label = cx.graph:node_label(loop_nid)
  local block_cx = cx:new_graph_scope(loop_label.block)

  return block_cx.graph:traverse_nodes_recursive(
    function(graph, nid, label)
      local inner_cx = block_cx:new_graph_scope(graph)
      return is_leaf(inner_cx, nid) or nil
    end)
end

local function is_task(cx, nid)
  local label = cx.graph:node_label(nid)
  return label:is(flow.node.Task)
end

local function task_has_privileges(cx, nid)
  -- Edges that are ok:
  --   * Happens-before
  --   * None
  --   * Reads on scalars, constants, or functions
  --   * Writes to fresh scalars
  -- Everything else is bad.

  local function is_bad_edge(edge)
    if edge.label:is(flow.edge.Read) then
      local label = cx.graph:node_label(edge.from_node)
      return not (label:is(flow.node.data.Scalar) or
                    label:is(flow.node.Constant) or
                    label:is(flow.node.Function))
    end
    if edge.label:is(flow.edge.Write) then
      local label = cx.graph:node_label(edge.to_node)
      return not (label:is(flow.node.data.Scalar) and label.fresh)
    end
    return not (edge.label:is(flow.edge.HappensBefore) or
                  edge.label:is(flow.edge.None))
  end

  return #cx.graph:filter_immediate_predecessors_by_edges(is_bad_edge, nid) > 0 or
    #cx.graph:filter_immediate_successors_by_edges(is_bad_edge, nid) > 0
end

local function nonleaf_tasks_no_privileges(cx, loop_nid)
  local loop_label = cx.graph:node_label(loop_nid)
  local block_cx = cx:new_graph_scope(loop_label.block)

  local result = block_cx.graph:traverse_nodes_recursive(
    function(graph, nid, label)
      local inner_cx = block_cx:new_graph_scope(graph)
      if is_task(inner_cx, nid) then
        return task_has_privileges(inner_cx, nid) and {graph, nid} or nil
      end
    end,
    function(graph, nid, label)
      local inner_cx = block_cx:new_graph_scope(graph)
      return not is_leaf(inner_cx, nid)
    end)
  if result == nil then
    return true
  else
    return false, unpack(result)
  end
end

local function filter_join(list1, list2, fn)
  local result = terralib.newlist()
  for _, elt1 in ipairs(list1) do
    for _, elt2 in ipairs(list2) do
      if fn(elt1, elt2) then
        result:insert(data.newtuple(elt1, elt2))
      end
    end
  end
  return result
end

local function get_incoming_reduction_op(cx, nid)
  local op = false
  local port = false
  cx.graph:traverse_incoming_edges(
    function(_, _, edge)
      if is_reduce(edge) then
        assert(not op or op == edge.label.op)
        op = edge.label.op
        port = edge.from_port
      end
    end,
    nid)
  return op, port
end

local function is_parallel_loop(cx, loop_nid)
  --  Check that all pairs of data nodes are non-interfering.

  local loop_label = cx.graph:node_label(loop_nid)
  local block_cx = cx:new_graph_scope(loop_label.block)

  local function can_interfere(nid1, nid2)
    if nid1 == nid2 then return false end -- Skip self checks.

    -- Check fields.
    local label1 = block_cx.graph:node_label(nid1)
    local label2 = block_cx.graph:node_label(nid2)
    if label1.field_path ~= label2.field_path then return false end

    -- Check for conflicts: read-write, read-reduce, etc.
    local read1 = #block_cx.graph:outgoing_read_set(nid1) > 0
    local read2 = #block_cx.graph:outgoing_read_set(nid2) > 0
    local write1 = #block_cx.graph:incoming_write_set(nid1) > 0
    local write2 = #block_cx.graph:incoming_write_set(nid2) > 0
    local reduce1 = get_incoming_reduction_op(block_cx, nid1)
    local reduce2 = get_incoming_reduction_op(block_cx, nid2)
    if not ((write1 and (read2 or write2 or reduce2)) or
            (write2 and (read1 or write1 or reduce1)) or
            (reduce1 and (read2 or (reduce2 and reduce2 ~= reduce1))) or
            (reduce2 and (read1 or (reduce1 and reduce1 ~= reduce2))))
    then
      return false
    end

    -- Check for region aliasing. (Aliased? Then it interferes.)
    local region1, region2 = label1.region_type, label2.region_type
    if not std.type_eq(region1, region2) then
      return std.type_maybe_eq(region1:fspace(), region2:fspace()) and
        block_cx.tree:can_alias(region1, region2)
    end

    -- Check for region indexing.
    if block_cx.tree:has_region_index(region1) then
      -- Is it indexed by the loop variable? (No? Then it interferes.)
      local index = block_cx.tree:region_index(region1)
      if not (
        (index:is(ast.typed.expr.ID) and index.value == loop_label.symbol) or
          (index:is(ast.typed.expr.Cast) and
             index.arg:is(ast.typed.expr.ID) and
             index.arg.value == loop_label.symbol))
      then
        return true
      end

      -- Is the parent disjoint? (No? Then it interferes.)
      local parent = block_cx.tree:parent(region1)
      if block_cx.tree:aliased(parent) then
        return true
      end
    end

    -- Fall through: no interference.
    return false
  end

  local data = block_cx.graph:filter_nodes(
    function(_, label) return label:is(flow.node.data) end)
  return #filter_join(data, data, can_interfere) == 0
end

local function leaves_are_parallel_loops(cx, loop_nid)
  local loop_label = cx.graph:node_label(loop_nid)
  local block_cx = cx:new_graph_scope(loop_label.block)

  local result = block_cx.graph:traverse_nodes_recursive(
    function(graph, nid, label)
      local inner_cx = block_cx:new_graph_scope(graph)
      if is_leaf(inner_cx, nid) then
        return not is_parallel_loop(inner_cx, nid) and {graph, nid} or nil
      end
    end)
  if result == nil then
    return true
  else
    return false, unpack(result)
  end
end

local function get_input(inputs, i, optional)
  if rawget(inputs, i) and #inputs[i] == 1 then
    return inputs[i][1].from_node, inputs[i][1]
  end
  assert(optional)
end

local function bounds_eq(label1, label2)
  if label1:is(flow.node.Constant) and label2:is(flow.node.Constant) then
    return label1.value.value == label2.value.value
  elseif label1:is(flow.node.data.Scalar) and label2:is(flow.node.data.Scalar) then
    return label1.region_type == label2.region_type and
      label1.field_path == label2.field_path
  end
end

local function is_exact_match(label, region_type, field_path)
  return label:is(flow.node.data) and
    label.region_type == region_type and
    (not field_path or label.field_path == field_path)
end

local function only(list)
  assert(#list == 1)
  return list[1]
end

local function maybe(list)
  return list[1]
end

local function first(list)
  assert(#list >= 1)
  return list[1]
end

local function find_first_instance(cx, value_label, optional)
  local function match(_, label)
    return is_exact_match(
      label, value_label.region_type, value_label.field_path)
  end
  return (optional and maybe or first)(cx.graph:filter_toposort(nil, match))
end

local function find_last_instance(cx, value_label, optional)
  local function match(_, label)
    return is_exact_match(
      label, value_label.region_type, value_label.field_path)
  end
  return (optional and maybe or first)(
    cx.graph:filter_inverse_toposort(nil, match))
end

-- Returns labels for the loop's bounds as a Terra list, consisting of two
-- labels for a ForNum loop (lower, upper) or one label for a ForList loop.
local function find_bounds_labels(cx, loop_nid, loop_label)
  local inputs = cx.graph:incoming_edges_by_port(loop_nid)
  if loop_label:is(flow.node.ctrl.ForNum) then
    return terralib.newlist({  -- Lower and upper bounds.
      cx.graph:node_label(get_input(inputs, 1)),
      cx.graph:node_label(get_input(inputs, 2)),
    })
  elseif loop_label:is(flow.node.ctrl.ForList) then
    return terralib.newlist({  -- The ispace/region/list iterated over.
      cx.graph:node_label(get_input(inputs, 1)),
    })
  else
    assert(false)
  end
end

local function loops_are_compatible(cx, loop_nid)
  local loop_label = cx.graph:node_label(loop_nid)
  local block_cx = cx:new_graph_scope(loop_label.block)

  -- Check that no ForNum leaves use strides.
  local check = block_cx.graph:traverse_nodes_recursive(
    function(graph, nid, label)
      local inner_cx = block_cx:new_graph_scope(graph)
      if is_leaf(inner_cx, nid) then
        local inputs = inner_cx.graph:incoming_edges_by_port(nid)
        return (label:is(flow.node.ctrl.ForNum) and get_input(inputs, 3, true)) or nil
      end
    end) == nil
  if not check then return false end

  -- Check that leaves use compatible bounds.
  local bounds_labels
  local check = block_cx.graph:traverse_nodes_recursive(
    function(graph, nid, label)
      local inner_cx = block_cx:new_graph_scope(graph)
      if is_leaf(inner_cx, nid) then
        local curr_bounds_labels = find_bounds_labels(inner_cx, nid, label)

        if not bounds_labels then
          bounds_labels = curr_bounds_labels
        else
          local labels_equal
          if #bounds_labels ~= #curr_bounds_labels then
            -- Loops should either all be ForNum or all be ForList, i.e. have same number of
            -- bounds.
            labels_equal = false
          else
            labels_equal = true
            for i = 1, #bounds_labels do
              labels_equal = labels_equal and bounds_eq(bounds_labels[i], curr_bounds_labels[i])
            end
          end
          return labels_equal and nil
        end
      end
    end) == nil
  if not check then return false end
  assert(bounds_labels)

  -- For Forlist loops, the bound should be a Scalar ispace.
  if (#bounds_labels == 1) then
    local forlist_bound = bounds_labels[1]
    if not forlist_bound:is(flow.node.data.Scalar) then return false end
    local bound_type = std.as_read(forlist_bound.value.expr_type)
    if not std.is_ispace(bound_type) then return false end
  end

  -- Check that bounds are never written.
  for _, bound_label in ipairs(bounds_labels) do
    local nid = bound_label:is(flow.node.data) and find_last_instance(block_cx, bound_label)
    if nid and #block_cx.graph:incoming_mutate_set(nid) > 0 then return false end
  end
  return true
end

local function ignore(...) end

local function can_spmdize(cx, loop)
  -- Conditions:
  --
  --  1. Loop has __demand(__spmd) (for now).
  --  2. Loop header is side-effect free. (Not applicable to for loops.)
  --  3. Loop body contains:
  --      a. For loops with:
  --           * Same index space
  --           * Parallelizable (no data dependencies)
  --      b. Open and close ops
  --      c. Data nodes (regions, partitions, scalars) of any kinds
  --      d. Any variables updated must be updated uniformly (not checked yet)
  --  4. Tasks outside of loops do not request privileges on regions.

  local has_demand = has_demand_spmd(cx, loop)

  local report_fail = ignore
  if has_demand then
    report_fail = report.error
  end

  local function fixup_label(label)
    if label:is(flow.node.Opaque) then return label.action end
    return label
  end

  -- Currently, only apply SPMD when explicitly requested.
  if not has_demand then
    local label = fixup_label(cx.graph:node_label(loop))
    report_fail(
      label,
      "unable to apply SPMD transformation: not requested")
    return false
  end

  do
    local ok, bad_graph, bad_nid = whitelist_node_types(cx, loop)
    if not ok then
      local label = fixup_label(bad_graph:node_label(bad_nid))
      report_fail(
        label,
        "unable to apply SPMD transformation: block contains bad node types")
      return false
    end
  end

  do
    local ok, bad_graph, bad_nid = nonleaf_tasks_no_privileges(cx, loop)
    if not ok then
      local label = fixup_label(bad_graph:node_label(bad_nid))
      report_fail(
        label,
        "unable to apply SPMD transformation: a task outside of a leaf loop requests privileges on a region")
      return false
    end
  end

  if not has_leaves(cx, loop) then
    local label = fixup_label(cx.graph:node_label(loop))
    report_fail(
      label,
      "unable to apply SPMD transformation: block contains no recognizable leaf loops")
    return false
  end

  do
    local ok, bad_graph, bad_nid = leaves_are_parallel_loops(cx, loop)
    if not ok then
      local label = fixup_label(bad_graph:node_label(bad_nid))
      report_fail(
        label,
        "unable to apply SPMD transformation: leaf loop is not parallelizable")
      return false
    end
  end

  if not loops_are_compatible(cx, loop) then
    local label = fixup_label(cx.graph:node_label(loop))
    report_fail(
      label,
      "unable to apply SPMD transformation: leaf loops use inconsistent bounds")
    return false
  end

  return true
end

local function find_open_results(cx, open_nid)
  local result_nids = terralib.newlist()
  cx.graph:traverse_immediate_successors(
    function(nid, label)
      if label:is(flow.node.data) then
        -- Include external nodes only, i.e. ones read or writen by
        -- something other than an open or close node.
        local is_external = false
        cx.graph:traverse_incoming_edges(
          function(from_node, _, edge)
            local label = cx.graph:node_label(from_node)
            if not (label:is(flow.node.Open) or label:is(flow.node.Close)) and
              isnt_happens_before(edge)
            then
              is_external = true
            end
          end,
          nid)
        cx.graph:traverse_outgoing_edges(
          function(_, to_node, edge)
            local label = cx.graph:node_label(to_node)
            if not (label:is(flow.node.Open) or label:is(flow.node.Close)) and
              isnt_happens_before(edge)
            then
              is_external = true
            end
          end,
          nid)
        if is_external then
          result_nids:insert(nid)
        end

        -- Recurse connected subgraph of open nodes.
        cx.graph:traverse_immediate_successors(
          function(nid, label)
            if label:is(flow.node.Open) then
              result_nids:insertall(find_open_results(cx, nid))
            end
          end,
          nid)
      end
    end,
    open_nid)
  return result_nids
end

local function find_close_results(cx, close_nid)
  local result_nids = terralib.newlist()
  cx.graph:traverse_immediate_successors(
    function(nid, label)
      if label:is(flow.node.data) then
        cx.graph:traverse_immediate_successors(
          function(nid, label)
            if label:is(flow.node.Open) then
              result_nids:insertall(find_open_results(cx, nid))
            end
          end,
          nid)
      end
    end,
    close_nid)
  return result_nids
end

local function find_matching_input(cx, op_nid, region_type, field_path)
  return maybe(cx.graph:filter_immediate_predecessors_by_edges(
    function(edge, label)
      return is_read_write(edge) and
        is_exact_match(label, region_type, field_path)
    end,
    op_nid))
end

local function find_matching_inputs(cx, op_nid, region_type, field_path)
  return cx.graph:filter_immediate_predecessors_by_edges(
    function(edge, label)
      return is_read_write(edge) and
        is_exact_match(label, region_type, field_path)
    end,
    op_nid)
end

local function find_matching_output(cx, op_nid, region_type, field_path)
  return maybe(cx.graph:filter_immediate_successors_by_edges(
    function(edge, label)
      return is_read_write(edge) and
        is_exact_match(label, region_type, field_path)
    end,
    op_nid))
end

local function find_match_backwards(cx, op_nid, region_type, field_path)
  return maybe(cx.graph:filter_nodes(
    function(nid, label)
      return is_exact_match(label, region_type, field_path) and
        cx.graph:reachable(nid, op_nid, is_read_write)
    end))
end

local function find_predecessor_maybe(cx, nid)
  local preds = cx.graph:filter_immediate_predecessors_by_edges(
    is_read_write, nid)
  if #preds == 0 then
    return
  end
  assert(#preds == 1)
  return preds[1]
end

local function find_open_source_close(cx, open_nid)
  local source_nid = find_predecessor_maybe(cx, open_nid)
  if not source_nid then return end
  assert(cx.graph:node_label(source_nid):is(flow.node.data))
  local close_nid = find_predecessor_maybe(cx, source_nid)
  if not close_nid then return end
  assert(cx.graph:node_label(close_nid):is(flow.node.Close))
  return close_nid
end

local function make_variable_label(cx, var_symbol, var_type, span)
  local node = ast.typed.expr.ID {
    value = var_symbol,
    expr_type = std.rawref(&var_type),
    annotations = ast.default_annotations(),
    span = span,
  }
  local var_region = cx.tree:intern_variable(
    node.expr_type, node.value, node.annotations, node.span)
  return flow.node.data.Scalar {
    value = node,
    region_type = var_region,
    field_path = data.newtuple(),
    fresh = false,
  }
end

local function make_constant(value, value_type, span)
  return ast.typed.expr.Constant {
    value = value,
    expr_type = value_type,
    annotations = ast.default_annotations(),
    span = span,
  }
end

local function compute_version_numbers(cx)
  local versions = data.newmap()
  local nids = cx.graph:toposort()
  for _, nid in ipairs(nids) do
    local edges = cx.graph:incoming_edges(nid)
    local version = 0
    for _, edge in ipairs(edges) do
      local contribute = 0
      local pred_label = cx.graph:node_label(edge.from_node)

      -- Writes immediately bump the version.
      if is_write(edge) and
        not (pred_label:is(flow.node.Open) or pred_label:is(flow.node.Close))
      then
        contribute = 1

      -- Reductions are pending until a subsequent read.
      elseif is_read(edge) and
        #cx.graph:filter_immediate_predecessors_by_edges(
          is_reduce, edge.from_node) > 0
      then
        contribute = 1
      end

      -- Record the maximum version (after contributions).
      if isnt_happens_before(edge) then
        version = data.max(version, versions[edge.from_node] + contribute)
      end
    end
    versions[nid] = version
  end
  return versions
end

local function compute_contributions(cx)
  local contributions = data.new_recursive_map(2)

  local nids = cx.graph:filter_toposort(
    nil, function(_, label) return label:is(flow.node.data) end)
  for _, nid in ipairs(nids) do
    local writers = cx.graph:filter_immediate_predecessors_by_edges(
      is_write_reduce, nid)
    if #writers > 0 then
      for _, writer_nid in ipairs(writers) do
        local writer_label = cx.graph:node_label(writer_nid)
        if writer_label:is(flow.node.Close) then
          -- This version gathers contributions from the close's inputs.
          local reads = cx.graph:filter_immediate_predecessors_by_edges(
            is_read, writer_nid)
          for _, other_nid in ipairs(reads) do
            for _, other_contribution in contributions[other_nid]:keys() do
              contributions[nid][other_contribution] = true
            end
          end
        else
          -- This version actually contains fresh data; reset contributions.
          contributions[nid][nid] = true
        end
      end
    end
  end

  return contributions
end

-- Returns the leaf loops' type (ForList or ForNum).
-- Assumes that all leaf nodes are loops of the same type.
local function find_leaf_loop_type(cx, loop_nid)
  local loop_label = cx.graph:node_label(loop_nid)
  local block_cx = cx:new_graph_scope(loop_label.block)

  return block_cx.graph:traverse_nodes_recursive(
    function(graph, nid, label)
      local inner_cx = cx:new_graph_scope(graph)
      if is_leaf(inner_cx, nid) then
        return label:type()
      end
    end)
end

local function normalize_communication_subgraph(cx, shard_loop)
  -- This step simplifies and normalizes the communication graph,
  -- removing opens and instances of parent regions. Close nodes in
  -- the resulting graph represent locations where explicit copies are
  -- required.
  --
  --  1. Normalize close inputs (to ensure all partitions exist at version 0).
  --  2. Remove opens (and regions used as intermediates).
  --  3. Remove trivial closes (consumed only by other closes).
  --  4. Normalize close outputs (remove spurious closes and close-outputs).
  --  5. Normalize final state (to ensure consistent final versions).
  --  6. Add reduction self-closes (for reductions involved in communication).
  --  7. Prune edges from read-reduce conflicts to avoid spurious copies.
  --  8. Fix up outer context to avoid naming intermediate regions.

  local shard_label = cx.graph:node_label(shard_loop)
  local block_cx = cx:new_graph_scope(shard_label.block)

  -- Normalize close inputs.
  local close_nids = block_cx.graph:filter_nodes(
    function(nid, label) return label:is(flow.node.Close) end)
  for _, close_nid in ipairs(close_nids) do
    -- Look forward in the graph to find regions resulting from this close.
    local result_nids = find_close_results(block_cx, close_nid)

    -- For each region, look back in the graph to find a matching region.
    local detach_nids = terralib.newlist()
    for _, result_nid in ipairs(result_nids) do
      local result_label = block_cx.graph:node_label(result_nid)
      -- Look for an exact match first.
      local input_nid = find_matching_input(
        block_cx, close_nid, result_label.region_type, result_label.field_path)
      if not input_nid then
        -- Otherwise look for the region elsewhere in the
        -- backwards-traversable graph. There are two cases:
        --
        --  1. If the region doesn't exist *at all*, just create it.
        --  2. If the region does exist, hook up to the appropriate node.

        -- Find the immediate ancestor.
        local ancestor_nids = block_cx.graph:filter_immediate_predecessors_by_edges(
          function(edge, label)
            return isnt_happens_before(edge) and
              block_cx.tree:is_subregion(result_label.region_type, label.region_type) and
              result_label.field_path == label.field_path
          end,
          close_nid)

        -- Keep the lowest among the immediate ancestors.
        local ancestor_nid
        do
          local lowest_nid
          for _, nid in ipairs(ancestor_nids) do
            if lowest_nid == nil then
              lowest_nid = nid
            elseif block_cx.tree:is_subregion(
              block_cx.graph:node_label(nid).region_type,
              block_cx.graph:node_label(lowest_nid).region_type)
            then
              lowest_nid = nid
            elseif not block_cx.tree:is_subregion(
              block_cx.graph:node_label(lowest_nid).region_type,
              block_cx.graph:node_label(nid).region_type)
            then
              assert(false, "incomparable regions")
            end
          end
          ancestor_nid = lowest_nid
        end
        assert(ancestor_nid)

        local any_input_nid = find_match_backwards(
          block_cx, close_nid, result_label.region_type, result_label.field_path)
        if not any_input_nid then
          -- 1. Region doesn't exist; just create it.
          input_nid = block_cx.graph:add_node(result_label)
          block_cx.graph:copy_outgoing_edges(
            function(edge) return edge.to_node == close_nid end, ancestor_nid, input_nid)
          detach_nids:insert(ancestor_nid)
        else
          -- 2. Region exists, somewhere back in the graph.

          -- This gets hard. For now, just pattern match cases we know
          -- how to handle and assert if we hit a more general case.

          -- Try to find the region among the ancestor's sources, if any.
          local ancestor_close_nid = find_predecessor_maybe(block_cx, ancestor_nid)
          if ancestor_close_nid then
            assert(block_cx.graph:node_label(ancestor_close_nid):is(flow.node.Close))

            local input_nid = first(block_cx.graph:filter_inverse_toposort(
              ancestor_close_nid,
              function(_, label)
                return is_exact_match(
                  label, result_label.region_type, result_label.field_path)
              end,
              is_read_write))

            assert(input_nid)
            block_cx.graph:add_edge(
              flow.edge.Read(flow.default_mode()), input_nid, block_cx.graph:node_result_port(input_nid),
              close_nid, block_cx.graph:node_available_port(close_nid))
          else
            assert(false)
          end
        end
      end
    end

    for _, nid in ipairs(detach_nids) do
      block_cx.graph:remove_outgoing_edges(
        function(edge) return edge.to_node == close_nid end, nid)
    end
  end

  -- Remove opens.
  local open_nids = block_cx.graph:filter_nodes(
    function(nid, label) return label:is(flow.node.Open) end)
  for _, open_nid in ipairs(open_nids) do
    local close_nid = find_open_source_close(block_cx, open_nid)
    if close_nid then
      local result_nids = block_cx.graph:immediate_successors(open_nid)
      for _, result_nid in ipairs(result_nids) do
        assert(block_cx.graph:node_label(result_nid):is(flow.node.data))
        block_cx.graph:replace_edges(result_nid, open_nid, close_nid)
      end
    end
    local source_nids = block_cx.graph:immediate_predecessors(open_nid)
    for _, source_nid in ipairs(source_nids) do
      block_cx.graph:remove_node(source_nid)
    end
    block_cx.graph:remove_node(open_nid)
  end

  -- Remove trivial closes.
  for _, close_nid in ipairs(close_nids) do
    -- Trivial closes are ones where the output of the close is
    -- immediately consumed only by close nodes.
    local has_reduction
    local source_nids = block_cx.graph:filter_immediate_predecessors_by_edges(
      isnt_happens_before, close_nid)
    for _, source_nid in ipairs(source_nids) do
      has_reduction = #block_cx.graph:filter_immediate_predecessors_by_edges(
        is_reduce, source_nid) > 0
      if has_reduction then break end
    end

    local has_nonclose, has_close
    local result_nids = block_cx.graph:filter_immediate_successors_by_edges(
      isnt_happens_before, close_nid)
    for _, result_nid in ipairs(result_nids) do
      local consumer_nids =
        block_cx.graph:filter_immediate_successors_by_edges(
          isnt_happens_before, result_nid)
      for _, consumer_nid in ipairs(consumer_nids) do
        if block_cx.graph:node_label(consumer_nid):is(flow.node.Close) then
          has_close = true
        end
        if not block_cx.graph:node_label(consumer_nid):is(flow.node.Close) then
          has_nonclose = true
        end
        if has_close and has_nonclose then break end
      end
      if has_close and has_nonclose then break end
    end

    local is_trivial = has_close and not has_nonclose and not has_reduction
    if is_trivial then
      for _, result_nid in ipairs(result_nids) do
        local result_label = block_cx.graph:node_label(result_nid)

        -- Copy incoming edges from result to matching source. (These
        -- should only be reductions.)
        do
          local source_nid = find_matching_input(
            block_cx, close_nid, result_label.region_type, result_label.field_path)

          if source_nid then
            block_cx.graph:copy_incoming_edges(
              function(edge)
                return isnt_happens_before(edge) and
                edge.from_node ~= close_nid
              end,
              result_nid, source_nid, false)
          end
        end

        -- Copy outgoing edges from result to all sources. (Remember,
        -- all consumers are close ops.)
        for _, source_nid in ipairs(source_nids) do
          block_cx.graph:copy_outgoing_edges(
            isnt_happens_before,
            result_nid, source_nid, false)
        end

        block_cx.graph:remove_node(result_nid)
      end
      block_cx.graph:remove_node(close_nid)
    end
  end

  -- Normalize close outputs.
  local versions = compute_version_numbers(block_cx)
  local close_nids = block_cx.graph:filter_nodes(
    function(nid, label) return label:is(flow.node.Close) end)
  for _, close_nid in ipairs(close_nids) do
    local result_nids = block_cx.graph:immediate_successors(close_nid)
    local versions_changed = 0
    for _, result_nid in ipairs(result_nids) do
      local result_label = block_cx.graph:node_label(result_nid)
      local input_nid = find_matching_input(
        block_cx, close_nid, result_label.region_type, result_label.field_path)
      assert(input_nid)
      if versions[result_nid] > versions[input_nid] then
        versions_changed = versions_changed + 1
      end
    end

    -- If multiple versions changed, split the close.
    if versions_changed > 1 then
      local contributor_nids = terralib.newlist()
      local input_nids = block_cx.graph:immediate_predecessors(close_nid)
      for _, input_nid in ipairs(input_nids) do
        local input_label = block_cx.graph:node_label(input_nid)
        if #block_cx.graph:filter_immediate_predecessors_by_edges(
            function(edge)
              local label = block_cx.graph:node_label(edge.from_node)
              return is_write_reduce(edge) and
                not (label:is(flow.node.Open) or label:is(flow.node.Close))
            end,
            input_nid) > 0
        then
          contributor_nids:insert(input_nid)
        end
      end
      assert(#contributor_nids > 0)

      local last_split
      for _, result_nid in ipairs(result_nids) do
        local result_label = block_cx.graph:node_label(result_nid)
        local input_nid = find_matching_input(
          block_cx, close_nid, result_label.region_type, result_label.field_path)
        assert(input_nid)
        if versions[result_nid] > versions[input_nid] then
          -- Version changed, count this as a split.
          if last_split then
            local split = block_cx.graph:add_node(flow.node.Close {})
            for _, nid in ipairs(contributor_nids) do
              block_cx.graph:copy_incoming_edges(
                function(edge) return edge.from_node == nid end, close_nid, split)
            end
            block_cx.graph:copy_incoming_edges(
              function(edge) return edge.from_node == input_nid end, close_nid, split, true)
            block_cx.graph:copy_outgoing_edges(
              function(edge) return edge.to_node == result_nid end, close_nid, split, true)
            block_cx.graph:add_edge(
              flow.edge.HappensBefore {}, last_split, block_cx.graph:node_sync_port(last_split),
              split, block_cx.graph:node_sync_port(split))
            last_split = split
          else
            last_split = close_nid
          end
        else
          -- Version didn't change, prune it back to the input.
          block_cx.graph:copy_outgoing_edges(function() return true end, result_nid, input_nid)
          block_cx.graph:remove_node(result_nid)
        end
      end

    -- If exactly one version changed, just prune back the regions that didn't change.
    elseif versions_changed == 1 then
      for _, result_nid in ipairs(result_nids) do
        local result_label = block_cx.graph:node_label(result_nid)
        local input_nid = find_matching_input(
          block_cx, close_nid, result_label.region_type, result_label.field_path)
        assert(input_nid)
        if versions[result_nid] <= versions[input_nid] then
          -- Version didn't change, prune it back to the input.
          block_cx.graph:copy_outgoing_edges(function() return true end, result_nid, input_nid)
          block_cx.graph:remove_node(result_nid)
        end
      end

    -- If no versions changed, remove the close entirely.
    else
      assert(versions_changed == 0)
      -- Bump each region to the next available version.
      local delete = true
      local input_nids = block_cx.graph:immediate_predecessors(close_nid)
      for _, input_nid in ipairs(input_nids) do
        local input_label = block_cx.graph:node_label(input_nid)
        local result_nid = find_matching_output(
          block_cx, close_nid, input_label.region_type, input_label.field_path)
        if result_nid then
          -- Don't destroy a close that was needed to separate reductions.
          if #block_cx.graph:filter_immediate_predecessors_by_edges(
            is_reduce, result_nid) > 0
          then
            delete = false
          else
            block_cx.graph:copy_outgoing_edges(function() return true end, result_nid, input_nid)
            block_cx.graph:remove_node(result_nid)
          end
        end
      end
      if delete then
        block_cx.graph:remove_node(close_nid)
      end
    end
  end

  -- Normalize final state.

  -- This is the step that captures loop-carried dependencies. Any
  -- data which is not up-to-date with any data it aliases is
  -- forceably updated at the end of the loop to ensure that the final
  -- state is consistent.

  -- Compute the most recent versions for each region.
  local data_nids = block_cx.graph:filter_nodes(
    function(_, label) return label:is(flow.node.data) end)
  local region_nids = data.new_recursive_map(1)
  local region_versions = data.new_recursive_map(1)
  for _, data_nid in ipairs(data_nids) do
    local label = block_cx.graph:node_label(data_nid)
    local region_type = label.region_type
    local field_path = label.field_path
    if not region_versions[region_type][field_path] or
      versions[data_nid] > region_versions[region_type][field_path]
    then
      region_nids[region_type][field_path] = data_nid
      region_versions[region_type][field_path] = versions[data_nid]
    end
  end

  -- Compute the contributions to each version of a region.
  local contributions = compute_contributions(block_cx)

  -- Check for and update out-of-date data.
  local previous_close_nids = block_cx.graph:filter_nodes(function(_, label) return label:is(flow.node.Close) end)
  for region_type, r1 in region_nids:items() do
    for field_path, _ in r1:items() do
      local newer_regions = terralib.newlist()
      if #block_cx.graph:outgoing_read_set(region_nids[region_type][field_path]) > 0 then
        for other_region_type, r2 in region_nids:items() do
          for other_field_path, _ in r2:items() do
            if not std.type_eq(region_type, other_region_type) and
              std.type_maybe_eq(region_type:fspace(), other_region_type:fspace()) and
              block_cx.tree:can_alias(region_type, other_region_type) and
              field_path == other_field_path and
              region_versions[region_type][field_path] < region_versions[other_region_type][field_path]
            then
              newer_regions:insert(other_region_type)
            end
          end
        end
      end

      -- The version check above is naive in the sense that it counts
      -- regions as changed even if they have only been updated by
      -- other copies.
      --
      -- Filter the set of newer versions to "fresh" regions,
      -- i.e. ones which have received direct updates on their own,
      -- and thus potentially contain data not stored elsewhere.
      do
        local fresh_regions = terralib.newlist()
        for _, newer_region in ipairs(newer_regions) do
          local newer_nid = region_nids[newer_region][field_path]
          local unique_contributions = contributions[newer_nid]:copy()

          for _, other_region in ipairs(newer_regions) do
            if not std.type_eq(newer_region, other_region) then
              local other_nid = region_nids[other_region][field_path]
              for _, other_contribution in contributions[other_nid]:keys() do
                local remove = terralib.newlist()
                for _, newer_contribution in unique_contributions:keys() do
                  local other_contribution_region = block_cx.graph:node_label(other_contribution).region_type
                  local newer_contribution_region = block_cx.graph:node_label(newer_contribution).region_type
                  if std.type_eq(other_contribution_region, newer_contribution_region) and
                    (versions[other_contribution] > versions[newer_contribution] or
                       (versions[other_contribution] == versions[newer_contribution] and
                          block_cx.graph:reachable(other_nid, newer_nid)))
                  then
                    remove:insert(newer_contribution)
                  end
                end
                for _, contribution in ipairs(remove) do
                  unique_contributions[contribution] = false
                end
              end
            end
          end

          local fresh = false
          for contribution, contribution_fresh in unique_contributions:items() do
            if contribution_fresh and versions[contribution] > region_versions[region_type][field_path] then
              fresh = true
              break
            end
          end
          if fresh then
            fresh_regions:insert(newer_region)
          end
        end
        newer_regions = fresh_regions
      end

      -- Apply updates in the original order.
      newer_regions:sort(
        function(a, b) return region_versions[a][field_path] < region_versions[b][field_path] end)

      if #newer_regions > 0 then
        local current_nid = region_nids[region_type][field_path]
        local current_label = block_cx.graph:node_label(current_nid)
        local first_nid = find_first_instance(block_cx, current_label)

        local reads = data.filter(
          function(nid) return not block_cx.graph:node_label(nid):is(flow.node.Close) end,
          block_cx.graph:outgoing_read_set(first_nid))
        if #reads > 0 then
          for _, newer_region_type in ipairs(newer_regions) do
            local close_nid = block_cx.graph:add_node(flow.node.Close {})
            local next_nid = block_cx.graph:add_node(current_label)
            local other_nid = region_nids[newer_region_type][field_path]

            block_cx.graph:add_edge(
              flow.edge.Read(flow.default_mode()),
              current_nid, block_cx.graph:node_result_port(current_nid),
              close_nid, 1)
            block_cx.graph:add_edge(
              flow.edge.Write(flow.default_mode()),
              close_nid, 1,
              next_nid, block_cx.graph:node_available_port(next_nid))
            block_cx.graph:add_edge(
              flow.edge.Read(flow.default_mode()),
              other_nid, block_cx.graph:node_result_port(other_nid),
              close_nid, 2)

            for _, nid in ipairs(previous_close_nids) do
              block_cx.graph:add_edge(
                flow.edge.HappensBefore {},
                nid, block_cx.graph:node_sync_port(nid),
                close_nid, block_cx.graph:node_sync_port(close_nid))
            end
            previous_close_nids:insert(close_nid)

            current_nid = next_nid
          end
        else
          print("FIXME: Skipping update of region which is not read")
        end
      end
    end
  end

  -- Add reduction self-closes (for reductions which will be communicated).
  local close_nids = block_cx.graph:filter_nodes(
    function(nid, label) return label:is(flow.node.Close) end)
  for _, close_nid in ipairs(close_nids) do
    local read_nids = block_cx.graph:incoming_read_set(close_nid)
    local write_nid = only(block_cx.graph:outgoing_write_set(close_nid))
    local write_label = block_cx.graph:node_label(write_nid)

    for _, read_nid in ipairs(read_nids) do
      local read_label = block_cx.graph:node_label(read_nid)
      if read_label.region_type ~= write_label.region_type then
        local reducers = block_cx.graph:filter_immediate_predecessors_by_edges(
          is_reduce, read_nid)
        if #reducers > 0 then
          local first_nid = find_first_instance(block_cx, read_label)
          local readers = data.filter(
            function(nid) return not block_cx.graph:node_label(nid):is(flow.node.Close) end,
            block_cx.graph:outgoing_read_set(first_nid))
          if #readers > 0 then
            local self_close_nid = block_cx.graph:add_node(flow.node.Close {})
            block_cx.graph:add_edge(
              flow.edge.HappensBefore {}, close_nid, block_cx.graph:node_sync_port(close_nid),
              self_close_nid, block_cx.graph:node_sync_port(self_close_nid))

            local final_nid = block_cx.graph:add_node(read_label)
            block_cx.graph:add_edge(
              flow.edge.Read(flow.default_mode()),
              read_nid, block_cx.graph:node_result_port(read_nid),
              self_close_nid, 1)
            block_cx.graph:add_edge(
              flow.edge.Write(flow.default_mode()), self_close_nid, 1,
              final_nid, block_cx.graph:node_result_port(final_nid))

            local outgoing = block_cx.graph:outgoing_edges(read_nid)
            for _, edge in ipairs(outgoing) do
              if edge.to_node ~= close_nid and edge.to_node ~= self_close_nid then
                block_cx.graph:add_edge(
                  edge.label, final_nid, edge.from_port,
                  edge.to_node, edge.to_port)
                block_cx.graph:remove_edge(
                  edge.from_node, edge.from_port, edge.to_node, edge.to_port)
              end
            end
          else
            print("FIXME: Skipping self-copy of region which is not read")
          end
        end
      end
    end
  end

  -- Prune edges from read-reduce conflicts to avoid spurious copies.
  data_nids = block_cx.graph:filter_nodes(
    function(_, label) return label:is(flow.node.data) end)
  for _, data_nid in ipairs(data_nids) do
    local close_nids = block_cx.graph:filter_immediate_successors_by_edges(
      function(edge)
        return is_read(edge) and
          block_cx.graph:node_label(edge.to_node):is(flow.node.Close)
      end,
    data_nid)
    local writer_nids = block_cx.graph:filter_immediate_predecessors_by_edges(
      is_write_reduce, data_nid)
    if #close_nids > 1 and #writer_nids == 0 then
      -- Look for a dominator among the close ops.
      local dominator_nid
      for _, nid in ipairs(close_nids) do
        local dominates = true
        for _, other_nid in ipairs(close_nids) do
          if not block_cx.graph:reachable(other_nid, nid, is_read_write) then
            dominates = false
            break
          end
        end
        if dominates then
          dominator_nid = nid
          break
        end
      end

      block_cx.graph:remove_outgoing_edges(
        function(edge)
          return is_read(edge) and
            edge.to_node ~= dominator_nid and
            block_cx.graph:node_label(edge.to_node):is(flow.node.Close)
        end,
        data_nid)
    end
  end

  -- At this point, the graph should have no regions.
  -- FIXME: Handle singleton regions.
  assert(not block_cx.graph:find_node(
           function(_, label) return label:is(flow.node.data.Region) end))

  -- Replace regions in the outer context as well.
  local region_nids = terralib.newlist()
  region_nids:insertall(
    cx.graph:filter_immediate_predecessors(
      function(_, label) return label:is(flow.node.data.Region) end,
      shard_loop))
  region_nids:insertall(
    cx.graph:filter_immediate_successors(
      function(_, label) return label:is(flow.node.data.Region) end,
      shard_loop))
  for _, region_nid in ipairs(region_nids) do
    local region_label = cx.graph:node_label(region_nid)

    -- Find the labels from the inner context that will replace this region.
    local replacement_labels = data.newmap()
    block_cx.graph:traverse_nodes(
      function(_, label)
        if label:is(flow.node.data) and
          std.type_maybe_eq(label.region_type:fspace(), region_label.region_type:fspace()) and
          cx.tree:can_alias(label.region_type, region_label.region_type) and
          label.field_path == region_label.field_path
        then
          replacement_labels[label.region_type] = label
        end
      end)
    assert(not replacement_labels:is_empty())

    local replacement_nids = terralib.newlist()
    for _, replacement_label in replacement_labels:items() do
      -- If there are an existing node, find the port it is attached to.
      local similar_nid = cx.graph:find_immediate_predecessor(
        function(_, label)
          return label:is(flow.node.data) and
            label.region_type == replacement_label.region_type
        end,
        shard_loop) or cx.graph:find_immediate_successor(
        function(_, label)
          return label:is(flow.node.data) and
            label.region_type == replacement_label.region_type
        end,
        shard_loop)
      local port
      if similar_nid then
        for _, edge in ipairs(cx.graph:incoming_edges(similar_nid)) do
          if edge.from_node == shard_loop then
            port = edge.from_port
            break
          end
        end
        if not port then
          for _, edge in ipairs(cx.graph:outgoing_edges(similar_nid)) do
            if edge.to_node == shard_loop then
              port = edge.to_port
              break
            end
          end
        end
      end
      if not port then
        port = cx.graph:node_available_port(shard_loop)
      end

      -- Build the replacement node and copy old edges.
      local nid = cx.graph:add_node(replacement_label)
      replacement_nids:insert(nid)
      cx.graph:copy_edges(shard_loop, region_nid, nid, port)
    end

    -- Add open and/or close nodes as necessary.
    local needs_open = false
    for _, edge in ipairs(cx.graph:incoming_edges(region_nid)) do
      if edge.from_node ~= shard_loop then
        needs_open = true
        break
      end
    end
    local needs_close = false
    for _, edge in ipairs(cx.graph:outgoing_edges(region_nid)) do
      if edge.to_node ~= shard_loop then
        needs_close = true
        break
      end
    end
    if needs_open or needs_close then
      if needs_open then
        local open_nid = cx.graph:add_node(flow.node.Open {})
        cx.graph:add_edge(
          flow.edge.Read(flow.default_mode()), region_nid, cx.graph:node_result_port(region_nid),
          open_nid, cx.graph:node_available_port(open_nid))
        for _, nid in ipairs(replacement_nids) do
          cx.graph:add_edge(
            flow.edge.Write(flow.default_mode()),
            open_nid, cx.graph:node_available_port(open_nid),
            nid, cx.graph:node_available_port(open_nid))
        end
      end
      if needs_close then
        local close_nid = cx.graph:add_node(flow.node.Close {})
        for _, nid in ipairs(replacement_nids) do
          cx.graph:add_edge(
            flow.edge.Read(flow.default_mode()),
            nid, cx.graph:node_result_port(nid),
            close_nid, cx.graph:node_available_port(close_nid))
        end
        cx.graph:add_edge(
          flow.edge.Write(flow.default_mode()),
          close_nid, cx.graph:node_result_port(region_nid),
          region_nid, cx.graph:node_available_port(region_nid))
      end

      cx.graph:remove_incoming_edges(
        function(edge) return edge.from_node == shard_loop end,
        region_nid)
      cx.graph:remove_outgoing_edges(
        function(edge) return edge.to_node == shard_loop end,
        region_nid)
    else
      cx.graph:remove_node(region_nid)
    end
  end

  assert(not block_cx.graph:find_node(
           function(_, label) return label:is(flow.node.data.Region) end))
end

local function normalize_communication(cx)
  -- Normalization is applied recursively to subgraphs, bottom-up.
  cx.graph:traverse_nodes_recursive(
    function(graph, nid, label)
      local inner_cx = cx:new_graph_scope(graph)
      if is_inner(inner_cx, nid) then
        normalize_communication_subgraph(inner_cx, nid)
      end
    end)
end

local function rewrite_shard_partitions(cx)
  -- Every partition (accessed to obtain a region) is replaced by a
  -- fresh list of regions, and the accessed region is replaced with a
  -- fresh region.
  --
  -- Every cross-product (accessed to obtain a partition) is replaced
  -- by a fresh list of partitions, and the accessed partition is
  -- replaced by a fresh partition.
  --
  -- Note: This means that you need to look where a node comes from to
  -- determine what it should be replaced by (a partition may or may
  -- not be replaced by a list).

  local function partition_is_cross_product_access(cx, value_type)
    local is_access = false
    cx.graph:traverse_nodes(
      function(nid, label)
        if label:is(flow.node.data) and label.region_type == value_type then
          local names = cx.graph:incoming_name_set(nid)
          is_access = is_access or (#names > 0)
        end
    end)
    return is_access
  end

  local function get_label_type(value_type)
    if std.is_region(value_type) then
      return flow.node.data.Region
    elseif std.is_partition(value_type) then
      return flow.node.data.Partition
    elseif std.is_list_of_regions(value_type) then
      return flow.node.data.List
    elseif std.is_list_of_partitions(value_type) then
      return flow.node.data.List
    else
      assert(false)
    end
  end

  local function make_fresh_type(cx, value_type, span)
    if std.is_region(value_type) then
      -- Hack: I'm going to cheat here and return the original region.
      return value_type
    elseif std.is_partition(value_type) then
      if partition_is_cross_product_access(cx, value_type) then
        -- Hack: I'm going to cheat here and return the original partition.
        return value_type
      else
        local region_type = value_type:parent_region()
        local expr_type = std.list(
          std.region(
            std.newsymbol(std.ispace(region_type:ispace().index_type)),
            region_type:fspace()),
          value_type)
        for other_region, _ in pairs(cx.tree.region_universe) do
          assert(not std.type_eq(expr_type, other_region))
          -- Only record explicit disjointness when there is possible
          -- type-based aliasing.
          if std.type_maybe_eq(expr_type:fspace(), other_region:fspace()) then
            std.add_constraint(cx.tree, expr_type, other_region, std.disjointness, true)
          end
        end
        cx.tree:intern_region_expr(expr_type, ast.default_annotations(), span)
        return expr_type
      end
    elseif std.is_cross_product(value_type) then
      local partition_type = value_type:subpartition_dynamic()
      local expr_type = std.list(partition_type)
      return expr_type
    else
      assert(false)
    end
  end

  local mapping = {}
  local symbols = data.newmap()
  local old_labels = data.new_recursive_map(1)
  local new_labels = data.new_recursive_map(1)
  cx.graph:traverse_nodes_recursive(
    function(graph, nid, label)
      local graph_cx = cx:new_graph_scope(graph)
      if label:is(flow.node.data.Region) or label:is(flow.node.data.Partition) or
        label:is(flow.node.data.CrossProduct)
      then
        local region_type = label.region_type
        local new_region_type
        if not mapping[region_type] then
          new_region_type = make_fresh_type(graph_cx, region_type, label.value.span)
          mapping[region_type] = new_region_type
          symbols[region_type] = std.newsymbol(
            mapping[region_type], "shard_" .. tostring(label.value.value))
        else
          new_region_type = mapping[region_type]
        end

        if not new_labels[mapping[region_type]][label.field_path] then
          -- Could use an AST rewriter here...
          local label_type = get_label_type(new_region_type)
          old_labels[region_type][label.field_path] = label
          new_labels[mapping[region_type]][label.field_path] = label_type(label) {
            region_type = mapping[region_type],
            value = label.value {
              value = symbols[region_type],
              expr_type = std.type_sub(label.value.expr_type, mapping),
            },
          }
        end
      end
    end)

  cx.graph:map_nodes_recursive(
    function(graph, nid, label)
      if label:is(flow.node.data.Region) or label:is(flow.node.data.Partition) or
        label:is(flow.node.data.CrossProduct)
      then
        assert(mapping[label.region_type])
        label = new_labels[mapping[label.region_type]][label.field_path]
      end
      assert(label)
      return label
    end)

  return new_labels, old_labels, mapping
end

local function rewrite_interior_regions(cx, old_type, new_type, new_symbol, make_fresh_type)
  local old_nids = cx.graph:filter_nodes(
    function(_, label)
      return label:is(flow.node.data) and label.region_type == old_type
    end)
  local mapping = { [old_type] = new_type }
  for _, old_nid in ipairs(old_nids) do
    local old_label = cx.graph:node_label(old_nid)
    local new_label = old_label {
      value = old_label.value {
        value = new_symbol,
        expr_type = std.type_sub(old_label.value.expr_type, mapping),
      },
      region_type = new_type,
    }
    local new_nid = cx.graph:add_node(new_label)
    cx.graph:replace_node(old_nid, new_nid)

    local index_nids = cx.graph:filter_immediate_successors_by_edges(
      function(edge, label)
        return is_none(edge) and label:is(flow.node.IndexAccess)
      end,
      new_nid)
    for _, index_nid in ipairs(index_nids) do
      local indexed_nids = cx.graph:outgoing_name_set(index_nid)
      assert(#indexed_nids >= 1)
      local old_indexed_label = cx.graph:node_label(indexed_nids[1])
      local old_indexed_type = old_indexed_label.region_type

      local new_indexed_type = make_fresh_type(cx, old_indexed_type, old_indexed_label.value.span)
      local new_indexed_symbol = std.newsymbol(new_indexed_type, "scratch_" .. tostring(old_indexed_label.value.value))
      rewrite_interior_regions(cx, old_indexed_type, new_indexed_type, new_indexed_symbol, make_fresh_type)
    end
  end
end

local function issue_with_scratch_fields(cx, op, reduction_nids, other_nids,
                                         user_nid, user_port,
                                         scratch_fields, mapping)
  local make_fresh_type = function(cx, value_type, span)
    if std.is_region(value_type) then
      local expr_type = std.region(
        std.newsymbol(std.ispace(value_type:ispace().index_type)),
        value_type:fspace())
      cx.tree:intern_region_expr(expr_type, ast.default_annotations(), span)
      return expr_type
    elseif std.is_list_of_regions(value_type) then
      local expr_type = value_type:slice()
      cx.tree:intern_region_expr(expr_type, ast.default_annotations(), span)
      return expr_type
    else
      assert(false)
    end
  end

  local old_label = cx.graph:node_label(reduction_nids[1])
  local old_type = old_label.region_type

  -- Apply scratch fields.
  local name_label = old_label.value
  local name_type = old_label.value.expr_type
  local fid_nids = terralib.newlist()
  for _, reduction_nid in ipairs(reduction_nids) do
    local field_path = cx.graph:node_label(reduction_nid).field_path

    local fid_label, fid_nid
    if scratch_fields[old_type][field_path] then
      fid_label, fid_nid = unpack(scratch_fields[old_type][field_path])
    else
      local fid_type = std.c.legion_field_id_t[1]
      local fid_symbol = std.newsymbol(fid_type, "fid_" .. tostring(old_label.value.value) .. tostring(field_path))
      fid_label = make_variable_label(cx, fid_symbol, fid_type, old_label.value.span)
      fid_nid = cx.graph:add_node(fid_label)
      scratch_fields[old_type][field_path] = data.newtuple(fid_label, fid_nid)
    end
    fid_nids:insert(fid_nid)

    name_type = make_fresh_type(cx, old_type, old_label.value.span)
    name_label = ast.typed.expr.WithScratchFields {
      region = ast.typed.expr.RegionRoot {
        region = name_label,
        fields = terralib.newlist({field_path}),
        expr_type = name_label.expr_type,
        annotations = old_label.value.annotations,
        span = old_label.value.span,
      },
      field_ids = fid_label.value,
      expr_type = name_type,
      annotations = ast.default_annotations(),
      span = old_label.value.span,
    }
  end

  -- Name the (with scratch fields) type.
  local new_type = name_type
  local new_symbol = std.newsymbol(new_type, "scratch_" .. tostring(old_label.value.value))
  mapping[old_type] = new_type

  name_label = flow.node.Opaque {
    action = ast.typed.stat.Var {
      symbols = terralib.newlist({new_symbol}),
      types = terralib.newlist({new_type}),
      values = terralib.newlist({name_label}),
      annotations = ast.default_annotations(),
      span = old_label.value.span,
    }
  }
  local name_nid = cx.graph:add_node(name_label)
  for _, nid in ipairs(reduction_nids) do
    cx.graph:add_edge(
      flow.edge.None(flow.default_mode()), nid, cx.graph:node_result_port(nid),
      name_nid, 1)
  end
  for _, nid in ipairs(other_nids) do
    cx.graph:add_edge(
      flow.edge.None(flow.default_mode()), nid, cx.graph:node_result_port(nid),
      name_nid, 1)
  end
  for _, nid in ipairs(fid_nids) do
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()), nid, cx.graph:node_result_port(nid),
      name_nid, 2)
  end

  -- Replicate normal inputs.
  for _, old_nid in ipairs(other_nids) do
    local old_label = cx.graph:node_label(old_nid)
    local new_label = old_label {
      value = old_label.value {
        value = new_symbol,
        expr_type = std.type_sub(old_label.value.expr_type, mapping),
      },
      region_type = new_type
    }
    local new_nid = cx.graph:add_node(new_label)

    cx.graph:add_edge(
      flow.edge.HappensBefore {}, name_nid, cx.graph:node_sync_port(name_nid),
      new_nid, cx.graph:node_sync_port(new_nid))

    -- Copy old edges.
    cx.graph:copy_outgoing_edges(
      function(edge)
        return is_read(edge) and
          edge.to_node == user_nid and edge.to_port == user_port
      end,
      old_nid, new_nid, true)
  end

  -- Replicate reduction inputs.
  for _, old_nid in ipairs(reduction_nids) do
    local old_label = cx.graph:node_label(old_nid)
    local field_path = old_label.field_path

    local new_label = old_label {
      value = old_label.value {
        value = new_symbol,
        expr_type = std.type_sub(old_label.value.expr_type, mapping),
      },
      region_type = new_type
    }
    local new_nid_cleared = cx.graph:add_node(new_label)
    local new_nid_reduced = cx.graph:add_node(new_label)

    cx.graph:add_edge(
      flow.edge.HappensBefore {}, name_nid, cx.graph:node_sync_port(name_nid),
      new_nid_cleared, cx.graph:node_sync_port(new_nid_cleared))

    -- Fill the new input.
    local init_type = std.get_field_path(old_type:fspace(), field_path)
    local init_value = std.reduction_op_init[op][init_type]
    assert(init_value)
    local init_label = flow.node.Constant {
      value = make_constant(init_value, init_type, old_label.value.span),
    }
    local init_nid = cx.graph:add_node(init_label)
    local fill_label = flow.node.Fill {
      dst_field_paths = terralib.newlist({field_path}),
      annotations = ast.default_annotations(),
      span = old_label.value.span,
    }
    local fill_nid = cx.graph:add_node(fill_label)
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()),
      new_nid_cleared, cx.graph:node_result_port(new_nid_cleared),
      fill_nid, 1)
    cx.graph:add_edge(
      flow.edge.Write(flow.default_mode()), fill_nid, 1, new_nid_reduced, 0)
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()), init_nid, cx.graph:node_result_port(init_nid),
      fill_nid, 2)

    -- Move reduction edges over to the new input.
    cx.graph:copy_incoming_edges(is_reduce, old_nid, new_nid_reduced, true)

    -- Move read edges over to new input.
    local copy_edges = cx.graph:copy_outgoing_edges(
      function(edge)
        return is_read(edge) and
          cx.graph:node_label(edge.to_node):is(flow.node.Copy) and
          #cx.graph:filter_outgoing_edges(
            function(edge2)
              local label = cx.graph:node_label(edge2.to_node)
              return is_write(edge2) and
                is_exact_match(label, old_type, field_path) and
                edge2.from_port == edge.to_port
            end,
            edge.to_node) == 0
      end,
      old_nid, new_nid_reduced, true)
  end

  -- Recursively apply the rewrite to nested control structures.
  local user_label = cx.graph:node_label(user_nid)
  if user_label:is(flow.node.ctrl.ForNum) then
    local block_cx = cx:new_graph_scope(user_label.block)
    rewrite_interior_regions(block_cx, old_type, new_type, new_symbol, make_fresh_type)
  end
end

local function issue_allocate_scratch_fields(cx, shard_loop, scratch_fields)
  for region_type, s1 in scratch_fields:items() do
    for field_path, fid in s1:items() do
      local fid_label, _ = unpack(fid)
      local region_nid = find_matching_input(cx, shard_loop, region_type, field_path)
      local region_label = cx.graph:node_label(region_nid)
      local create_label = flow.node.Opaque {
        action = ast.typed.stat.Var {
          symbols = terralib.newlist({fid_label.value.value}),
          types = terralib.newlist({std.as_read(fid_label.value.expr_type)}),
          values = terralib.newlist({
              ast.typed.expr.AllocateScratchFields {
                region = ast.typed.expr.RegionRoot {
                  region = region_label.value,
                  fields = terralib.newlist({field_path}),
                  expr_type = region_label.value.expr_type,
                  annotations = ast.default_annotations(),
                  span = region_label.value.span,
                },
                expr_type = std.as_read(fid_label.value.expr_type),
                annotations = ast.default_annotations(),
                span = fid_label.value.span,
              },
          }),
          annotations = ast.default_annotations(),
          span = fid_label.value.span,
        },
      }
      local create_nid = cx.graph:add_node(create_label)

      local fid_nid = find_first_instance(cx, fid_label)

      cx.graph:add_edge(
        flow.edge.HappensBefore {}, create_nid, cx.graph:node_sync_port(create_nid),
        fid_nid, cx.graph:node_available_port(fid_nid))

      cx.graph:add_edge(
        flow.edge.Read(flow.default_mode()), fid_nid, cx.graph:node_result_port(fid_nid),
        shard_loop, cx.graph:node_available_port(shard_loop))
    end
  end
end

local function is_reduction_communicated(cx, nid)
  local op = get_incoming_reduction_op(cx, nid)
  local copy_nids = cx.graph:filter_immediate_successors_by_edges(
    function(edge)
      return is_read(edge) and
        cx.graph:node_label(edge.to_node):is(flow.node.Copy)
    end,
    nid)
  return op and #copy_nids > 0
end

local function raise_scratch_fields(cx, loop_nid, scratch_fields)
  local loop_label = cx.graph:node_label(loop_nid)
  local block_cx = cx:new_graph_scope(loop_label.block)

  for _, s1 in scratch_fields:items() do
    for _, fid in s1:items() do
      local fid_label, _ = unpack(fid)
      if find_first_instance(block_cx, fid_label, true) then
        local nid = find_first_instance(cx, fid_label, true) or
          cx.graph:add_node(fid_label)
        local port = cx.graph:node_available_port(loop_nid)
        cx.graph:add_edge(
          flow.edge.Read(flow.default_mode()),
          nid, cx.graph:node_result_port(nid),
          loop_nid, port)
      end
    end
  end
end

local function rewrite_reduction_scratch_fields_subgraph(
    cx, nid, scratch_fields, mapping)
  local loop_label = cx.graph:node_label(nid)
  local block_cx = cx:new_graph_scope(loop_label.block)

  local data_nids = block_cx.graph:filter_nodes(
    function(_, label) return label:is(flow.node.data) end)
  local touched_nids = {}
  for _, data_nid in ipairs(data_nids) do
    if not touched_nids[data_nid] and is_reduction_communicated(block_cx, data_nid) then
      local op, port = get_incoming_reduction_op(block_cx, data_nid)
      local user_nid = only(block_cx.graph:filter_immediate_predecessors_by_edges(
        is_reduce, data_nid))
      local reduction_nids = block_cx.graph:filter_immediate_successors_by_edges(
        function(edge)
          return edge.from_port == port and
            is_reduction_communicated(block_cx, edge.to_node)
        end,
        user_nid)
      local other_nids = block_cx.graph:filter_immediate_predecessors_by_edges(
        function(edge)
          return edge.to_port == port and is_read(edge)
        end,
        user_nid)

      issue_with_scratch_fields(
        block_cx, op, reduction_nids, other_nids, user_nid, port,
        scratch_fields, mapping)

      for _, reduction_nid in ipairs(reduction_nids) do
        touched_nids[reduction_nid] = true
      end
    end
  end

  raise_scratch_fields(cx, nid, scratch_fields)
end

local function rewrite_reduction_scratch_fields(cx, shard_loop)
  -- Rewrite reduction copies to use scratch fields.
  local mapping = {}
  local scratch_fields = data.new_recursive_map(1)
  cx.graph:traverse_nodes_recursive(
    function(graph, nid)
      local inner_cx = cx:new_graph_scope(graph)
      if is_inner(inner_cx, nid) then
        rewrite_reduction_scratch_fields_subgraph(inner_cx, nid, scratch_fields, mapping)
      end
    end)

  -- Raise scratch field IDs to level of outer loop.
  issue_allocate_scratch_fields(cx, shard_loop, scratch_fields)

  -- Create a mapping to stop scratch fields from propogating.
  local scratch_field_mapping = {}
  for _, s1 in scratch_fields:items() do
    for _, fid in s1:items() do
      local fid_label, _ = unpack(fid)
      scratch_field_mapping[fid_label.region_type] = false
    end
  end

  return scratch_field_mapping
end

local function issue_self_copy(cx, src_nid, dst_in_nid, dst_out_nid, op)
  local src_label = cx.graph:node_label(src_nid)
  local dst_label = cx.graph:node_label(dst_in_nid)
  assert(src_label.field_path == dst_label.field_path)
  local field_path = src_label.field_path

  -- Add the copy.
  local field_paths = terralib.newlist({field_path})
  local copy = flow.node.Copy {
    src_field_paths = field_paths,
    dst_field_paths = field_paths,
    op = op,
    annotations = ast.default_annotations(),
    span = src_label.value.span,
  }
  local copy_nid = cx.graph:add_node(copy)

  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()), src_nid, cx.graph:node_result_port(src_nid),
    copy_nid, 1)
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    dst_in_nid, cx.graph:node_result_port(dst_in_nid),
    copy_nid, 2)
  cx.graph:add_edge(
    flow.edge.Write(flow.default_mode()), copy_nid, 2,
    dst_out_nid, cx.graph:node_available_port(dst_out_nid))

  return copy_nid
end

local function issue_intersection_copy(cx, src_nid, dst_in_nid, dst_out_nid, op,
                                       intersections)
  local src_label = cx.graph:node_label(src_nid)
  local src_type = src_label.region_type
  local dst_label = cx.graph:node_label(dst_in_nid)
  local dst_type = dst_label.region_type
  assert(src_label.field_path == dst_label.field_path)
  local field_path = src_label.field_path

  local intersection_label
  if intersections[src_type][dst_type][field_path] then
    intersection_label = intersections[src_type][dst_type][field_path]
  else
    for _, label in intersections[src_type][dst_type]:items() do
      intersection_label = label { field_path = field_path }
      break
    end
    if not intersection_label then
      local intersection_type = std.list(
        std.list(dst_type:subregion_dynamic(), nil, 1), nil, 1)
      -- FIXME: Should this be a subregion of the original list or its
      -- own root? The regions themselves do alias at runtime, so
      -- that's the rationale behind them being subregions, but I'm
      -- not sure all subsequent stages will handle this properly.
      std.add_constraint(cx.tree, intersection_type, dst_type, std.subregion, false)
      local intersection_symbol = std.newsymbol(
        intersection_type,
        "intersection_" .. tostring(src_label.value.value) .. "_" ..
          tostring(dst_label.value.value))
      intersection_label = dst_label {
        value = dst_label.value {
          value = intersection_symbol,
          expr_type = std.type_sub(dst_label.value.expr_type,
                                   {[dst_type] = intersection_type}),
        },
        region_type = intersection_type,
      }
    end
    intersections[src_type][dst_type][field_path] = intersection_label
  end
  local intersection_in_nid = cx.graph:add_node(intersection_label)
  local intersection_out_nid = cx.graph:add_node(intersection_label)

  -- Add happens-before synchronization on the intersection nids since
  -- these won't be constrained by the copy.
  cx.graph:add_edge(
    flow.edge.HappensBefore {},
    dst_in_nid, cx.graph:node_sync_port(dst_in_nid),
    intersection_in_nid, cx.graph:node_sync_port(intersection_in_nid))
  cx.graph:add_edge(
    flow.edge.HappensBefore {},
    intersection_out_nid, cx.graph:node_sync_port(intersection_out_nid),
    dst_out_nid, cx.graph:node_sync_port(dst_out_nid))

  -- Add the copy.
  local field_paths = terralib.newlist({field_path})
  local copy = flow.node.Copy {
    src_field_paths = field_paths,
    dst_field_paths = field_paths,
    op = op,
    annotations = ast.default_annotations(),
    span = src_label.value.span,
  }
  local copy_nid = cx.graph:add_node(copy)

  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()), src_nid, cx.graph:node_result_port(src_nid),
    copy_nid, 1)
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    intersection_in_nid, cx.graph:node_result_port(intersection_in_nid),
    copy_nid, 2)
  cx.graph:add_edge(
    flow.edge.Write(flow.default_mode()), copy_nid, 2,
    intersection_out_nid, cx.graph:node_available_port(intersection_out_nid))

  return copy_nid
end

local function issue_barrier_advance(cx, v0_nid)
  local v0 = cx.graph:node_label(v0_nid)
  local v1_nid = cx.graph:add_node(v0)

  local advance = flow.node.Advance {
    expr_type = std.as_read(v0.value.expr_type),
    annotations = ast.default_annotations(),
    span = v0.value.span,
  }
  local advance_nid = cx.graph:add_node(advance)
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()), v0_nid, cx.graph:node_result_port(v0_nid),
    advance_nid, 1)
  cx.graph:add_edge(
    flow.edge.Write(flow.default_mode()),
    advance_nid, cx.graph:node_result_port(advance_nid),
    v1_nid, cx.graph:node_available_port(v1_nid))
  return v1_nid
end

local function issue_barrier_preadvance(cx, v1_nid)
  local v1 = cx.graph:node_label(v1_nid)
  local v0_nid = cx.graph:add_node(v1)

  local advance = flow.node.Advance {
    expr_type = std.as_read(v1.value.expr_type),
    annotations = ast.default_annotations(),
    span = v1.value.span,
  }
  local advance_nid = cx.graph:add_node(advance)
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()), v0_nid, cx.graph:node_result_port(v0_nid),
    advance_nid, 1)
  cx.graph:add_edge(
    flow.edge.Write(flow.default_mode()),
    advance_nid, cx.graph:node_result_port(advance_nid),
    v1_nid, cx.graph:node_available_port(v1_nid))
  return v0_nid
end

local function issue_barrier_adjust(cx, barrier_nid, delta)
  local barrier_label = cx.graph:node_label(barrier_nid)
  local delta_label
  if type(delta) == "number" then
    delta_label = make_constant(delta, int, barrier_label.value.span)
  else
    delta_label = delta
  end

  local adjust_label = flow.node.Opaque {
    action = ast.typed.stat.Assignment {
      lhs = terralib.newlist({barrier_label.value}),
      rhs = terralib.newlist({
        ast.typed.expr.Adjust {
          barrier = barrier_label.value,
          value = delta_label,
          expr_type = barrier_label.value.expr_type,
          annotations = ast.default_annotations(),
          span = barrier_label.value.span,
        }
      }),
      annotations = ast.default_annotations(),
      span = barrier_label.value.span,
    }
  }
  local adjust_nid = cx.graph:add_node(adjust_label)
  cx.graph:add_edge(
    flow.edge.Arrive {}, adjust_nid, 1,
    barrier_nid, cx.graph:node_available_port(barrier_nid))
  return adjust_nid
end

local function index_phase_barriers(cx, loop_label, bar_list_label)
  -- Find the loop index for this loop.
  local index_nids = cx.graph:filter_nodes(
    function(_, label)
      return label:is(flow.node.data.Scalar) and
        label.value.value == loop_label.symbol
    end)
  assert(#index_nids == 1)
  local index_nid = index_nids[1]
  local index_label = cx.graph:node_label(index_nid)

  -- Select the barriers for this iteration.
  local bar_list_type = std.as_read(bar_list_label.value.expr_type)
  assert(std.is_list_of_phase_barriers(bar_list_type))
  local bar_type = bar_list_type.element_type
  local block_bar_list_nid = cx.graph:add_node(bar_list_label)

  local bar_symbol = std.newsymbol(bar_type, "bar_" .. tostring(bar_list_label.value.value))
  local bar_label = make_variable_label(
    cx, bar_symbol, bar_type, bar_list_label.value.span) { fresh = true }
  local block_bar_nid = cx.graph:add_node(bar_label)

  local block_index_bar_nid = cx.graph:add_node(
    flow.node.IndexAccess {
      expr_type = bar_type,
      annotations = ast.default_annotations(),
      span = bar_label.value.span,
    })
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    block_bar_list_nid, cx.graph:node_result_port(block_bar_list_nid),
    block_index_bar_nid, 1)
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    index_nid, cx.graph:node_result_port(index_nid),
    block_index_bar_nid, 2)

  cx.graph:add_edge(
    flow.edge.Write(flow.default_mode()),
    block_index_bar_nid, cx.graph:node_result_port(block_index_bar_nid),
    block_bar_nid, 0)

  return block_bar_nid
end

local issue_barrier_arrive_loop
local function issue_barrier_arrive(cx, bar_nid, use_nid)
  local use_label = cx.graph:node_label(use_nid)

  cx.graph:add_edge(
    flow.edge.Arrive {}, use_nid, cx.graph:node_available_port(use_nid),
    bar_nid, cx.graph:node_available_port(bar_nid))
  if use_label:is(flow.node.ctrl.ForNum) or use_label:is(flow.node.ctrl.ForList) then
    issue_barrier_arrive_loop(cx, bar_nid, use_nid)
  elseif not (use_label:is(flow.node.Task) or use_label:is(flow.node.Copy) or
                use_label:is(flow.node.Close))
  then
    print("FIXME: Issued arrive on " .. tostring(use_label:type()))
  end
end

function issue_barrier_arrive_loop(cx, bar_list_nid, use_nid)
  local use_label = cx.graph:node_label(use_nid)
  local block_cx = cx:new_graph_scope(use_label.block)

  local bar_list_label = cx.graph:node_label(bar_list_nid)
  local block_bar_nid = index_phase_barriers(block_cx, use_label, bar_list_label)

  -- Hack: The proper way to do this is probably to compute the
  -- frontier of operations at the end of the loop (or better,
  -- producers for the region we're protecting with the barrier, but
  -- that might be hard).
  local block_compute_nids = block_cx.graph:filter_nodes(
    function(_, label) return label:is(flow.node.Task) end)
  assert(#block_compute_nids == 1)
  local block_compute_nid = block_compute_nids[1]

  issue_barrier_arrive(block_cx, block_bar_nid, block_compute_nid)
end

local issue_barrier_await_loop
local function issue_barrier_await(cx, bar_nid, use_nid)
  local use_label = cx.graph:node_label(use_nid)

  cx.graph:add_edge(
    flow.edge.Await {},
    bar_nid, cx.graph:node_result_port(bar_nid),
    use_nid, cx.graph:node_available_port(use_nid))
  if use_label:is(flow.node.ctrl.ForNum) or use_label:is(flow.node.ctrl.ForList) then
    issue_barrier_await_loop(cx, bar_nid, use_nid)
  elseif not (use_label:is(flow.node.Task) or use_label:is(flow.node.Copy) or
                use_label:is(flow.node.Close))
  then
    print("FIXME: Issued arrive on " .. tostring(use_label:type()))
  end
end

function issue_barrier_await_loop(cx, bar_list_nid, use_nid)
  local use_label = cx.graph:node_label(use_nid)
  local block_cx = cx:new_graph_scope(use_label.block)

  local bar_list_label = cx.graph:node_label(bar_list_nid)
  local block_bar_nid = index_phase_barriers(block_cx, use_label, bar_list_label)

  -- Hack: The proper way to do this is probably to compute the
  -- frontier of operations at the start of the loop (or better,
  -- consumers for the region we're protecting with the barrier, but
  -- that might be hard).
  local block_compute_nids = block_cx.graph:filter_nodes(
    function(_, label) return label:is(flow.node.Task) end)
  assert(#block_compute_nids == 1)
  local block_compute_nid = block_compute_nids[1]

  issue_barrier_await(block_cx, block_bar_nid, block_compute_nid)
end

local generate_empty_task
do
  local empty_task
  function generate_empty_task(cx, span)
    if empty_task then
      return empty_task
    end

    local privileges = terralib.newlist()
    local coherence_modes = data.new_recursive_map(1)
    local flags = data.new_recursive_map(2)
    local conditions = {}
    local constraints = false
    local return_type = int

    local name = data.newtuple("empty_task_" .. tostring(std.newsymbol(return_type)))
    local prototype = std.newtask(name)
    local task_type = {} -> return_type
    prototype:settype(task_type)
    prototype:set_param_symbols(terralib.newlist())
    prototype:setprivileges(privileges)
    prototype:set_coherence_modes(coherence_modes)
    prototype:set_flags(flags)
    prototype:set_conditions(conditions)
    prototype:set_param_constraints(constraints)
    prototype:set_constraints(cx.tree.constraints)
    prototype:set_region_universe(cx.tree.region_universe)

    local ast = ast.typed.top.Task {
      name = name,
      params = terralib.newlist(),
      return_type = return_type,
      privileges = privileges,
      coherence_modes = coherence_modes,
      flags = flags,
      conditions = conditions,
      constraints = constraints,
      body = ast.typed.Block {
        stats = terralib.newlist({
            ast.typed.stat.Return {
              value = ast.typed.expr.Constant {
                value = 0,
                expr_type = return_type,
                annotations = ast.default_annotations(),
                span = span,
              },
              annotations = ast.default_annotations(),
              span = span,
            },
        }),
        span = span,
      },
      config_options = ast.TaskConfigOptions {
        leaf = false,
        inner = false,
        idempotent = false,
      },
      region_divergence = false,
      prototype = prototype,
      annotations = ast.default_annotations(),
      span = span,
    }

    -- Hack: Can't include passes here, because that would create a
    -- cyclic dependence. Name each optimization individually.

    -- passes.optimize(ast)
    if std.config["index-launch"] then ast = optimize_index_launches.entry(ast) end
    if std.config["future"] then ast = optimize_futures.entry(ast) end
    if std.config["leaf"] or std.config["inner"] then ast = optimize_config_options.entry(ast) end
    print("FIXME: Mapping optimization disabled while generating empty task")
    -- if std.config["mapping"] then ast = optimize_mapping.entry(ast) end
    if std.config["trace"] then ast = optimize_traces.entry(ast) end
    if std.config["no-dynamic-branches"] then ast = optimize_divergence.entry(ast) end
    if std.config["vectorize"] then ast = vectorize_loops.entry(ast) end

    if std.config["pretty"] then print(pretty.entry(ast)) end
    ast = codegen.entry(ast)

    empty_task = ast
    return ast
  end
end

local terra block_on_future(x : int) end

local function issue_barrier_await_blocking(cx, bar_nid, use_nid, after_nid, inner_sync_points)
  local use_label = cx.graph:node_label(use_nid)

  local var_nid
  if inner_sync_points[use_nid] then
    var_nid = inner_sync_points[use_nid]
  else
    local var_type = int
    local var_symbol = std.newsymbol(var_type, "inner_sync_point")
    local var_label = make_variable_label(cx, var_symbol, var_type, use_label.span)
    var_nid = cx.graph:add_node(var_label)
    inner_sync_points[use_nid] = var_nid

    local def_label = flow.node.Opaque {
      action = ast.typed.stat.Var {
        symbols = terralib.newlist({var_symbol}),
        values = terralib.newlist({
            ast.typed.expr.Constant {
              value = 0,
              expr_type = var_type,
              annotations = ast.default_annotations(),
              span = use_label.span,
            }
        }),
        types = terralib.newlist({var_type}),
        annotations = ast.default_annotations(),
        span = use_label.span,
      }
    }
    local def_nid = cx.graph:add_node(def_label)
    cx.graph:add_edge(
      flow.edge.HappensBefore {},
      def_nid, cx.graph:node_sync_port(def_nid),
      var_nid, cx.graph:node_sync_port(var_nid))

    -- Force execution to block by consuming the value.
    local sink_label = flow.node.Function {
      value = ast.typed.expr.Function {
        value = block_on_future,
        expr_type = block_on_future:gettype(),
        annotations = ast.default_annotations(),
        span = use_label.span,
      }
    }
    local sink_nid = cx.graph:add_node(sink_label)

    local consume_label = flow.node.Task {
      opaque = true,
      expr_type = terralib.types.unit,
      annotations = ast.default_annotations(),
      span = use_label.span,
    }
    local consume_nid = cx.graph:add_node(consume_label)

    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()),
      sink_nid, cx.graph:node_result_port(sink_nid),
      consume_nid, 1)
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()),
      var_nid, cx.graph:node_result_port(var_nid),
      consume_nid, 2)

    if not cx.graph:reachable(use_nid, after_nid) then
      cx.graph:add_edge(
        flow.edge.HappensBefore {},
        consume_nid, cx.graph:node_sync_port(consume_nid),
        use_nid, cx.graph:node_sync_port(use_nid))
    end
  end

  local empty_task = generate_empty_task(cx, use_label.span)

  local task_label = flow.node.Function {
    value = ast.typed.expr.Function {
      value = empty_task,
      expr_type = empty_task:get_type(),
      annotations = ast.default_annotations(),
      span = use_label.span,
    }
  }
  local task_nid = cx.graph:add_node(task_label)

  local scratch_type = int
  local scratch_symbol = std.newsymbol(scratch_type)
  local scratch_label = make_variable_label(cx, scratch_symbol, scratch_type, use_label.span){ fresh = true }
  local scratch_nid = cx.graph:add_node(scratch_label)

  local call_label = flow.node.Task {
    opaque = false,
    expr_type = scratch_type,
    annotations = ast.default_annotations(),
    span = use_label.span,
  }
  local call_nid = cx.graph:add_node(call_label)

  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    task_nid, cx.graph:node_result_port(task_nid),
    call_nid, 1)
  cx.graph:add_edge(
    flow.edge.Write(flow.default_mode()),
    call_nid, cx.graph:node_result_port(call_nid),
    scratch_nid, 0)

  local reduce_label = flow.node.Reduce {
    op = "+",
    annotations = ast.default_annotations(),
    span = use_label.span,
  }
  local reduce_nid = cx.graph:add_node(reduce_label)
  cx.graph:add_edge(
    flow.edge.Reduce({op = "+", coherence = flow.default_coherence(), flag = flow.default_flag()}),
    reduce_nid, 1,
    var_nid, cx.graph:node_result_port(var_nid))
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    scratch_nid, cx.graph:node_result_port(scratch_nid),
    reduce_nid, 2)

  issue_barrier_await(cx, bar_nid, call_nid)

  cx.graph:add_edge(
    flow.edge.HappensBefore {},
    after_nid, cx.graph:node_sync_port(after_nid),
    call_nid, cx.graph:node_sync_port(call_nid))
end

-- The phase barriers here are labeled full/empty * in/out * v0/vi/vi1
--
--   * Full means that the producer has finished writing, and the
--     data is now ready to be read by the consumer.
--   * Empty means that the consumer has finished reading the data,
--     and the producer can start writing the buffer with the next
--     set of data.
--
--   * Out refers to the producer and in refers to the consumer.
--
--   * Versions are determined as follows:
--       * v0 is the first occurance of the barrier (phase 0)
--       * vi is the current occurance (phase i)
--       * vi1 is the next occurance (phase i+1)
--       * vn is the last occurance (phase n)
--       * vn1 is beyond the existing last occurance (phase n+1)

local function issue_intersection_copy_synchronization_forwards(
    cx, src_nid, dst_in_nid, dst_out_nid, copy_nid, barriers, inner_sync_points)
  local src_label = cx.graph:node_label(src_nid)
  local dst_label = cx.graph:node_label(dst_in_nid)
  local dst_type = dst_label.region_type
  local field_path = dst_label.field_path

  local empty_in, empty_out, full_in, full_out
  local bars = barriers[dst_type][field_path]
  if bars then
    empty_in, empty_out, full_in, full_out = unpack(bars)
  else
    local bar_type = std.list(std.list(std.phase_barrier))
    -- Incoming barriers work differently when used with copy-copy
    -- synchronization. The depth parameter ensures that codegen
    -- issues the synchronization properly.
    local in_bar_type = std.list(
      std.list(std.phase_barrier, nil, nil, nil, nil, 1),
      nil, nil, nil, nil, 1)

    local suffix = tostring(src_label.value.value) .. "_" .. tostring(dst_label.value.value) .. "_" .. tostring(field_path)

    local empty_in_symbol = std.newsymbol(in_bar_type, "empty_in_" .. suffix)
    empty_in = make_variable_label(
      cx, empty_in_symbol, in_bar_type, dst_label.value.span)

    local empty_out_symbol = std.newsymbol(bar_type, "empty_out_" .. suffix)
    empty_out = make_variable_label(
      cx, empty_out_symbol, bar_type, dst_label.value.span)

    local full_in_symbol = std.newsymbol(in_bar_type, "full_in_" .. suffix)
    full_in = make_variable_label(
      cx, full_in_symbol, in_bar_type, dst_label.value.span)

    local full_out_symbol = std.newsymbol(bar_type, "full_out_" .. suffix)
    full_out = make_variable_label(
      cx, full_out_symbol, bar_type, dst_label.value.span)

    barriers[dst_type][field_path] = data.newtuple(
      empty_in, empty_out, full_in, full_out)
  end

  -- Find the consumer for the data.
  local consumer_wrapped = false
  local consumer_nids = cx.graph:outgoing_read_set(dst_out_nid)
  if #consumer_nids == 0 then
    -- Wrap around to top of graph.
    consumer_wrapped = true
    local initial_nid = find_first_instance(cx, dst_label)
    consumer_nids = cx.graph:outgoing_read_set(initial_nid)
    if #consumer_nids == 0 then
      consumer_nids = cx.graph:incoming_write_set(initial_nid)
    end
  end
  assert(#consumer_nids > 0)
  consumer_nids = data.filter(
    function(nid)
      local dominated = false
      for _, other_nid in ipairs(consumer_nids) do
        if nid ~= other_nid and cx.graph:reachable(
          other_nid, nid, is_read_write)
        then
          dominated = true
          break
        end
      end
      return not dominated
    end, consumer_nids)
  assert(#consumer_nids > 0)

  local function get_current_instance(cx, label)
    return find_last_instance(cx, label, true) or cx.graph:add_node(label)
  end

  -- Get the current phase.
  local full_in_vi_nid = get_current_instance(cx, full_in)
  local full_out_vi_nid = get_current_instance(cx, full_out)

  -- Get the next phase.
  local full_in_vi1_nid = issue_barrier_advance(cx, full_in_vi_nid)
  local full_out_vi1_nid = issue_barrier_advance(cx, full_out_vi_nid)

  -- Get the first phase (for wrap-around).
  local full_in_v0_nid = find_first_instance(cx, full_in)
  local full_out_v0_nid = find_first_instance(cx, full_out)

  local full_in_nid = (consumer_wrapped and full_in_v0_nid) or full_in_vi1_nid
  local full_out_nid = full_out_vi_nid

  issue_barrier_arrive(cx, full_out_nid, copy_nid)
  for _, consumer_nid in ipairs(consumer_nids) do
    if is_inner(cx, consumer_nid) then
      issue_barrier_await_blocking(cx, full_in_nid, consumer_nid, copy_nid, inner_sync_points)
    else
      issue_barrier_await(cx, full_in_nid, consumer_nid)
    end
  end
end

local function issue_intersection_copy_synchronization_backwards(
    cx, src_nid, dst_in_nid, dst_out_nid, copy_nid, barriers)
  local dst_label = cx.graph:node_label(dst_in_nid)
  local dst_type = dst_label.region_type
  local field_path = dst_label.field_path

  local bars = barriers[dst_type][field_path]
  assert(bars)
  local empty_in, empty_out, full_in, full_out = unpack(bars)

  -- Find the previous consumer for the data.
  local prev_consumer_wrapped = false
  local prev_consumer_nids = data.filter(
    function(nid) return not cx.graph:node_label(nid):is(flow.node.Close) end,
    cx.graph:outgoing_read_set(dst_in_nid))
  if #prev_consumer_nids == 0 then
    prev_consumer_nids = cx.graph:incoming_write_set(dst_in_nid)
  end
  if #prev_consumer_nids == 0 then
    -- Wrap around to top of graph.
    prev_consumer_wrapped = true
    local final_nid = find_last_instance(cx, dst_label)
    prev_consumer_nids = cx.graph:outgoing_read_set(final_nid)
    if #prev_consumer_nids == 0 then
      prev_consumer_nids = cx.graph:incoming_write_set(final_nid)
    end
  end
  -- If there are more than one of these, we need to increase the
  -- expected arrival count. Try to reduce the number of arrivers by
  -- filtering out backwards-dominated consumers.
  assert(#prev_consumer_nids > 0)
  prev_consumer_nids = data.filter(
    function(nid)
      local dominated = false
      for _, other_nid in ipairs(prev_consumer_nids) do
        if nid ~= other_nid and cx.graph:reachable(
          nid, other_nid, is_read_write)
        then
          dominated = true
          break
        end
      end
      return not dominated
    end,
    prev_consumer_nids)
  assert(#prev_consumer_nids > 0)

  -- Right now the synchronization will get messed up if any previous
  -- consumer is an inner loop, so just assert that it's not.
  for _, nid in ipairs(prev_consumer_nids) do
    assert(not is_inner(cx, nid), "previous consumer is inner loop")
  end

  local function get_current_instance(cx, label)
    return find_first_instance(cx, label, true) or cx.graph:add_node(label)
  end

  -- Get the next phase.
  local empty_in_vi1_nid = get_current_instance(cx, empty_in)
  local empty_out_vi1_nid = get_current_instance(cx, empty_out)

  -- Get the current phase.
  local empty_in_vi_nid = not prev_consumer_wrapped and
    issue_barrier_preadvance(cx, empty_in_vi1_nid)
  local empty_out_vi_nid = not prev_consumer_wrapped and
    issue_barrier_preadvance(cx, empty_out_vi1_nid)

  -- Get the last phase (for wrap-around).
  local empty_in_vn_nid = find_last_instance(cx, empty_in)
  local empty_in_vn1_nid = prev_consumer_wrapped and
    issue_barrier_advance(cx, empty_in_vn_nid)
  local empty_out_vn_nid = find_last_instance(cx, empty_out)
  local empty_out_vn1_nid = prev_consumer_wrapped and
    issue_barrier_advance(cx, empty_out_vn_nid)

  local empty_in_nid = (prev_consumer_wrapped and empty_in_vn_nid) or empty_in_vi_nid
  local empty_out_nid = empty_out_vi1_nid

  if #prev_consumer_nids > 1 then
    -- Adjust arrival count of in barrier.
    local adjust_nid = issue_barrier_adjust(
      cx, empty_in_nid, #prev_consumer_nids - 1)

    -- Make sure adjustment is sequenced properly with other operations.
    cx.graph:add_edge(
      flow.edge.HappensBefore {},
      dst_in_nid, cx.graph:node_sync_port(dst_in_nid),
      adjust_nid, cx.graph:node_sync_port(adjust_nid))
    for _, prev_consumer_nid in ipairs(prev_consumer_nids) do
      cx.graph:add_edge(
        flow.edge.HappensBefore {},
        adjust_nid, cx.graph:node_sync_port(adjust_nid),
        prev_consumer_nid, cx.graph:node_sync_port(prev_consumer_nid))
    end
  end

  issue_barrier_await(cx, empty_out_nid, copy_nid)
  for _, prev_consumer_nid in ipairs(prev_consumer_nids) do
    issue_barrier_arrive(cx, empty_in_nid, prev_consumer_nid)
  end
end

local function rewrite_copies_subgraph(cx, loop_nid, inverse_mapping, intersections)
  local loop_label = cx.graph:node_label(loop_nid)
  local block_cx = cx:new_graph_scope(loop_label.block)

  -- Issue copies for each close.
  local close_nids = block_cx.graph:filter_toposort(
    nil, function(_, label) return label:is(flow.node.Close) end)
  local remove = terralib.newlist()
  local sync = data.newmap()
  for _, close_nid in ipairs(close_nids) do
    local dst_out_nid = only(block_cx.graph:outgoing_write_set(close_nid))
    local dst_label = block_cx.graph:node_label(dst_out_nid)
    local dst_type = dst_label.region_type
    local dst_in_nid = find_matching_input(
      block_cx, close_nid, dst_label.region_type, dst_label.field_path)
    assert(dst_in_nid)

    local needs_removal = false
    local src_nids = block_cx.graph:incoming_read_set(close_nid)
    for _, src_nid in ipairs(src_nids) do
      local src_label = block_cx.graph:node_label(src_nid)
      local src_type = src_label.region_type

      local op = get_incoming_reduction_op(block_cx, src_nid)
      local reduction_closes = op and data.filter(
        function(nid) return block_cx.graph:node_label(nid):is(flow.node.Close) end,
        block_cx.graph:outgoing_read_set(src_nid))
      if src_nid ~= dst_in_nid or (reduction_closes and #reduction_closes > 1) then
        local copy_nid
        if src_type == dst_type and
          not block_cx.tree:aliased(inverse_mapping[src_type])
        then
          copy_nid = issue_self_copy(
            block_cx, src_nid, dst_in_nid, dst_out_nid, op)
        else
          copy_nid = issue_intersection_copy(
            block_cx, src_nid, dst_in_nid, dst_out_nid, op, intersections)
          sync[copy_nid] = data.newtuple(
            src_nid, dst_in_nid, dst_out_nid, src_type,
            dst_label.region_type, dst_label.field_path)
        end

        -- Migrate other types of edges...
        block_cx.graph:copy_incoming_edges(
          is_happens_before, close_nid, copy_nid, false)
        block_cx.graph:copy_outgoing_edges(
          is_happens_before, close_nid, copy_nid, false)

        needs_removal = true
      end
    end

    if needs_removal then
      remove:insert(close_nid)
    end
  end

  -- Remove obsolete closes.
  for _, close_nid in ipairs(remove) do
    block_cx.graph:remove_node(close_nid)
  end

  return sync
end

local function raise_barriers_forwards(cx, loop_nid, barriers)
  local loop_label = cx.graph:node_label(loop_nid)
  local block_cx = cx:new_graph_scope(loop_label.block)

  for _, b1 in barriers:items() do
    for _, b2 in b1:items() do
      for _, b3 in b2:items() do
        local empty_in, empty_out, full_in, full_out = unpack(b3)
        for _, barrier in ipairs({full_in, full_out}) do
          if find_first_instance(block_cx, barrier, true) then
            local last_nid = find_last_instance(cx, barrier, true) or
              cx.graph:add_node(barrier)
            local next_nid = cx.graph:add_node(barrier)
            local port = cx.graph:node_available_port(loop_nid)
            cx.graph:add_edge(
              flow.edge.Read(flow.default_mode()),
              last_nid, cx.graph:node_result_port(last_nid),
              loop_nid, port)
            cx.graph:add_edge(
              flow.edge.Write(flow.default_mode()),
              loop_nid, port,
              next_nid, cx.graph:node_available_port(next_nid))
          end
        end
      end
    end
  end
end

local function raise_barriers_backwards(cx, loop_nid, barriers)
  local loop_label = cx.graph:node_label(loop_nid)
  local block_cx = cx:new_graph_scope(loop_label.block)

  for _, b1 in barriers:items() do
    for _, b2 in b1:items() do
      for _, b3 in b2:items() do
        local empty_in, empty_out, full_in, full_out = unpack(b3)
        for _, barrier in ipairs({empty_in, empty_out}) do
          if find_first_instance(block_cx, barrier, true) then
            local first_nid = find_first_instance(cx, barrier, true) or
              cx.graph:add_node(barrier)
            local prev_nid = cx.graph:add_node(barrier)
            local port = cx.graph:node_available_port(loop_nid)
            cx.graph:add_edge(
              flow.edge.Read(flow.default_mode()),
              prev_nid, cx.graph:node_result_port(prev_nid),
              loop_nid, port)
            cx.graph:add_edge(
              flow.edge.Write(flow.default_mode()),
              loop_nid, port,
              first_nid, cx.graph:node_available_port(first_nid))
          end
        end
      end
    end
  end
end

local function rewrite_synchronization_forwards_subgraph(cx, loop_nid, sync, barriers, inner_sync_points)
  local loop_label = cx.graph:node_label(loop_nid)
  local block_cx = cx:new_graph_scope(loop_label.block)

  -- Issue copy synchronization in the forwards direction.
  local used_forwards = data.new_recursive_map(2)
  for _, nid in ipairs(block_cx.graph:toposort()) do
    if sync[nid] then
      local src_nid, dst_in_nid, dst_out_nid, src_type, dst_type, field_path = unpack(sync[nid])
      issue_intersection_copy_synchronization_forwards(
        block_cx, src_nid, dst_in_nid, dst_out_nid, nid, barriers[src_type], inner_sync_points)
    elseif is_inner(block_cx, nid) then
      raise_barriers_forwards(block_cx, nid, barriers)
    end
  end
end

local function rewrite_synchronization_backwards_subgraph(cx, loop_nid, sync, barriers)
  local loop_label = cx.graph:node_label(loop_nid)
  local block_cx = cx:new_graph_scope(loop_label.block)

  local used_backwards = data.new_recursive_map(2)
  for _, nid in ipairs(block_cx.graph:inverse_toposort()) do
    if sync[nid] then
    local src_nid, dst_in_nid, dst_out_nid, src_type, dst_type, field_path = unpack(sync[nid])
    issue_intersection_copy_synchronization_backwards(
      block_cx, src_nid, dst_in_nid, dst_out_nid, nid, barriers[src_type])
    elseif is_inner(block_cx, nid) then
      raise_barriers_backwards(block_cx, nid, barriers)
    end
  end
end

local function rewrite_communication(cx, shard_loop, mapping)
  -- Every close operation has zero or one pass-through inputs and
  -- zero or more copied inputs. Copied inputs come in two forms:
  -- blits and reductions.
  --
  -- The pass-through input (if any) is always the same type as the
  -- output, and is always (a) a reduction (b) which is not used in
  -- communication (i.e. it will not be transformed into reduction
  -- buffer later). Since the data (by definition) lives at the
  -- destination, there is no need to copy it.
  --
  -- Other inputs (if any) become copied inputs. Copied inputs are
  -- distinguished by whether the source region was the target of a
  -- reduction. If so, the copy is issued with the given reduction
  -- operator, otherwise, the copy is issued as a blit (no reduction
  -- operator).

  local shard_label = cx.graph:node_label(shard_loop)
  local block_cx = cx:new_graph_scope(shard_label.block)

  local inverse_mapping = {}
  for k, v in pairs(mapping) do
    inverse_mapping[v] = k
  end

  -- Issue copies for each close.
  local intersections = data.new_recursive_map(2)
  local sync = {}
  cx.graph:traverse_nodes_recursive(
    function(graph, nid, label)
      local inner_cx = cx:new_graph_scope(graph)
      if is_inner(inner_cx, nid) then
        sync[graph] = rewrite_copies_subgraph(inner_cx, nid, inverse_mapping, intersections)
      end
    end)

  -- Issue synchronization for each copy.
  local barriers = data.new_recursive_map(2)
  local inner_sync_points = {}
  cx.graph:traverse_nodes_recursive(
    function(graph, nid, label)
      if not inner_sync_points[graph] then inner_sync_points[graph] = data.newmap() end
      local inner_cx = cx:new_graph_scope(graph)
      if is_inner(inner_cx, nid) then
        rewrite_synchronization_forwards_subgraph(
          inner_cx, nid, sync[graph], barriers, inner_sync_points[graph])
      end
    end)
  cx.graph:traverse_nodes_recursive(
    function(graph, nid, label)
      local inner_cx = cx:new_graph_scope(graph)
      if is_inner(inner_cx, nid) then
        rewrite_synchronization_backwards_subgraph(inner_cx, nid, sync[graph], barriers)
      end
    end)

  -- Raise intersections as arguments to loop.
  for _, i1 in intersections:items() do
    for _, i2 in i1:items() do
      local port = cx.graph:node_available_port(shard_loop)
      for _, list in i2:items() do
        local in_nid = cx.graph:add_node(list)
        local out_nid = cx.graph:add_node(list)
        cx.graph:add_edge(
          flow.edge.Read(flow.default_mode()), in_nid, cx.graph:node_result_port(in_nid),
          shard_loop, port)
        cx.graph:add_edge(
          flow.edge.Write(flow.default_mode()), shard_loop, port,
          out_nid, cx.graph:node_available_port(out_nid))
      end
    end
  end

  -- Raise barriers as arguments to loop.
  for _, b1 in barriers:items() do
    for _, b2 in b1:items() do
      for _, b3 in b2:items() do
        for _, barrier in ipairs(b3) do
          local nid = cx.graph:add_node(barrier)
          cx.graph:add_edge(
            flow.edge.Read(flow.default_mode()), nid, cx.graph:node_result_port(nid),
            shard_loop, cx.graph:node_available_port(shard_loop))
        end
      end
    end
  end

  return intersections, barriers
end

local function rewrite_scalar_communication_subgraph(cx, loop_nid)
  local loop_label = cx.graph:node_label(loop_nid)
  local block_cx = cx:new_graph_scope(loop_label.block)

  -- We're going to do this with two collectives. First, local
  -- contributions will be summed into a local collective. Then that
  -- collective will be advanced and the result contributed to the
  -- global collective, which produces the final value. This avoids
  -- the creation of N tasks to compute the local contribution.

  local collectives = terralib.newlist()
  local local_collectives = terralib.newlist()

  local target_nids = cx.graph:filter_immediate_successors_by_edges(
    function(edge, label)
      return label:is(flow.node.data.Scalar) and is_reduce(edge)
    end,
    loop_nid)
  for _, target_nid in ipairs(target_nids) do
    local target_label = cx.graph:node_label(target_nid)
    local target_type = std.as_read(target_label.value.expr_type)

    local op = get_incoming_reduction_op(cx, target_nid)
    local collective_type = std.dynamic_collective(target_type)

    -- ###
    -- ### Interior of loop
    -- ###

    -- 0. Identify the local contribution.
    local block_target_nids = block_cx.graph:filter_nodes(
      function(nid, label)
        return is_exact_match(label, target_label.region_type, target_label.field_path)
        end)
    assert(#block_target_nids == 1) -- For now there must be exactly one.
    local block_target_nid = block_target_nids[1]

    local block_contributor_nids = block_cx.graph:filter_immediate_predecessors_by_edges(
      is_reduce, block_target_nid)
    assert(#block_contributor_nids == 1) -- For now there must be exactly one.
    local block_contributor_nid = block_contributor_nids[1]
    assert(block_cx.graph:node_label(block_contributor_nid):is(flow.node.Reduce))

    local block_contribution_nid = only(
      block_cx.graph:filter_immediate_predecessors_by_edges(
        is_read, block_contributor_nid))
    local block_contribution_label = block_cx.graph:node_label(block_contribution_nid)
    assert(block_contribution_label:is(flow.node.data.Scalar))

    -- 1. Create the local collective.
    local local_label = make_variable_label(
      cx, std.newsymbol(collective_type, "local_collective_" .. tostring(target_label.value.value)),
      collective_type, target_label.value.span)
    local block_local_nid = block_cx.graph:add_node(local_label)

    -- And remember it for later...
    local_collectives:insert(data.newtuple(local_label, op))

    -- 2. Arrive at the local collective with the local contribution.
    local block_arrive_label = flow.node.Opaque {
      action = ast.typed.stat.Expr {
        expr = ast.typed.expr.Arrive {
          barrier = local_label.value,
          value = block_contribution_label.value,
          expr_type = collective_type,
          annotations = ast.default_annotations(),
          span = target_label.value.span,
        },
        annotations = ast.default_annotations(),
        span = target_label.value.span,
      }
    }
    local block_arrive_nid = block_cx.graph:add_node(block_arrive_label)
    block_cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()),
      block_local_nid, block_cx.graph:node_result_port(block_local_nid),
      block_arrive_nid, block_cx.graph:node_available_port(block_arrive_nid))
    block_cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()),
      block_contribution_nid, block_cx.graph:node_result_port(block_contribution_nid),
      block_arrive_nid, block_cx.graph:node_available_port(block_arrive_nid))

    -- 3. Remove local references to the target.
    block_cx.graph:remove_node(block_contributor_nid)
    block_cx.graph:remove_node(block_target_nid)

    -- ###
    -- ### Exterior of loop
    -- ###

    -- Local collective versions for outside the loop.
    local local_v0_nid = cx.graph:add_node(local_label)
    local local_v1_nid = cx.graph:add_node(local_label)

    -- 4. Record arrival of loop on local collective.
    cx.graph:add_edge(
      flow.edge.Arrive {},
      loop_nid, cx.graph:node_available_port(loop_nid),
      local_v0_nid, cx.graph:node_available_port(local_v0_nid))

    -- 5. Advance the local collective.
    local local_advance_label = flow.node.Advance {
      expr_type = collective_type,
      annotations = ast.default_annotations(),
      span = target_label.value.span,
    }
    local local_advance_nid = cx.graph:add_node(local_advance_label)
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()),
      local_v0_nid, cx.graph:node_result_port(local_v0_nid),
      local_advance_nid, 1)
    cx.graph:add_edge(
      flow.edge.Write(flow.default_mode()),
      local_advance_nid, cx.graph:node_result_port(local_advance_nid),
      local_v1_nid, cx.graph:node_available_port(local_v1_nid))

    -- 6. Retrieve the local contribution value.
    local contribution_label = make_variable_label(
      cx, std.newsymbol(target_type, "local_" .. tostring(target_label.value.value)),
      target_type, target_label.value.span) { fresh = true }
    local contribution_nid = cx.graph:add_node(contribution_label)

    local local_result_label = flow.node.Opaque {
      action = ast.typed.expr.DynamicCollectiveGetResult {
        value = local_label.value,
        expr_type = target_type,
        annotations = ast.default_annotations(),
        span = target_label.value.span,
      }
    }
    local local_result_nid = cx.graph:add_node(local_result_label)
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()),
      local_v1_nid, cx.graph:node_result_port(local_v1_nid),
      local_result_nid, cx.graph:node_available_port(local_result_nid))
    cx.graph:add_edge(
      flow.edge.Write(flow.default_mode()),
      local_result_nid, cx.graph:node_result_port(local_result_nid),
      contribution_nid, 0)

    -- 7. Create the global collective.
    local collective_label = make_variable_label(
      cx, std.newsymbol(collective_type, "collective_" .. tostring(target_label.value.value)),
      collective_type, target_label.value.span)
    local collective_v0_nid = cx.graph:add_node(collective_label)
    local collective_v1_nid = cx.graph:add_node(collective_label)

    -- And remember it for later...
    collectives:insert(data.newtuple(collective_label, op))

    -- 8. Arrive at the collective with the local contribution value.
    local arrive_label = flow.node.Opaque {
      action = ast.typed.stat.Expr {
        expr = ast.typed.expr.Arrive {
          barrier = collective_label.value,
          value = contribution_label.value,
          expr_type = collective_type,
          annotations = ast.default_annotations(),
          span = target_label.value.span,
        },
        annotations = ast.default_annotations(),
        span = target_label.value.span,
      }
    }
    local arrive_nid = cx.graph:add_node(arrive_label)
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()),
      collective_v0_nid, cx.graph:node_result_port(collective_v0_nid),
      arrive_nid, cx.graph:node_available_port(arrive_nid))
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()),
      contribution_nid, cx.graph:node_result_port(contribution_nid),
      arrive_nid, cx.graph:node_available_port(arrive_nid))

    -- 9. Advance the collective.
    local advance_label = flow.node.Advance {
      expr_type = collective_type,
      annotations = ast.default_annotations(),
      span = target_label.value.span,
    }
    local advance_nid = cx.graph:add_node(advance_label)
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()),
      collective_v0_nid, cx.graph:node_result_port(collective_v0_nid),
      advance_nid, 1)
    cx.graph:add_edge(
      flow.edge.Write(flow.default_mode()),
      advance_nid, cx.graph:node_result_port(advance_nid),
      collective_v1_nid, cx.graph:node_available_port(collective_v1_nid))

    cx.graph:add_edge(
      flow.edge.HappensBefore {},
      arrive_nid, cx.graph:node_sync_port(arrive_nid),
      advance_nid, cx.graph:node_sync_port(advance_nid))

    -- 10. Retrieve the global contribution and reduce into the target.
    local global_label = make_variable_label(
      cx, std.newsymbol(target_type, "global_" .. tostring(target_label.value.value)),
      target_type, target_label.value.span) { fresh = true }
    local global_nid = cx.graph:add_node(global_label)

    local result_label = flow.node.Opaque {
      action = ast.typed.expr.DynamicCollectiveGetResult {
        value = collective_label.value,
        expr_type = target_type,
        annotations = ast.default_annotations(),
        span = target_label.value.span,
      }
    }
    local result_nid = cx.graph:add_node(result_label)
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()),
      collective_v1_nid, cx.graph:node_result_port(collective_v1_nid),
      result_nid, cx.graph:node_available_port(result_nid))
    cx.graph:add_edge(
      flow.edge.Write(flow.default_mode()),
      result_nid, cx.graph:node_result_port(result_nid),
      global_nid, 0)

    local reduce_label = flow.node.Reduce {
      op = op,
      annotations = ast.default_annotations(),
      span = target_label.value.span,
    }
    local reduce_nid = cx.graph:add_node(reduce_label)
    cx.graph:add_edge(
      flow.edge.Reduce({
          op = op,
          coherence = flow.default_coherence(),
          flag = flow.default_flag(),
      }),
      reduce_nid, 1,
      target_nid, cx.graph:node_available_port(target_nid))
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()),
      global_nid, cx.graph:node_result_port(global_nid),
      reduce_nid, 2)

    -- 11. Remove reduction edges from target.
    cx.graph:remove_incoming_edges(
      function(edge) return is_reduce(edge) and edge.from_node == loop_nid end,
      target_nid)
  end

  return collectives, local_collectives
end

local function raise_collectives(cx, loop_nid, collectives)
  local loop_label = cx.graph:node_label(loop_nid)
  local block_cx = cx:new_graph_scope(loop_label.block)

  for _, c in ipairs(collectives) do
    local collective, op = unpack(c)
    if find_first_instance(block_cx, collective, true) then
      local collective_nid = cx.graph:add_node(collective)
      cx.graph:add_edge(
        flow.edge.Read(flow.default_mode()),
        collective_nid, cx.graph:node_result_port(collective_nid),
        loop_nid, cx.graph:node_available_port(loop_nid))
    end
  end
end

local function rewrite_scalar_communication(cx)
  local collectives = terralib.newlist()
  local local_collectives = terralib.newlist()
  cx.graph:traverse_nodes_recursive(
    function(graph, nid, label)
      local inner_cx = cx:new_graph_scope(graph)
      if is_leaf(inner_cx, nid) then
        local cs, local_cs = rewrite_scalar_communication_subgraph(
          inner_cx, nid)
        collectives:insertall(cs)
        local_collectives:insertall(local_cs)
      end
    end)
  cx.graph:traverse_nodes_recursive(
    function(graph, nid, label)
      local inner_cx = cx:new_graph_scope(graph)
      if is_inner(inner_cx, nid) then
        raise_collectives(inner_cx, nid, collectives)
        raise_collectives(inner_cx, nid, local_collectives)
      end
    end)

  return collectives, local_collectives
end

local function issue_local_collective_creation(cx, loop_nid, collective_nid, op,
                                               nparts_nid)
  local collective = cx.graph:node_label(collective_nid)
  local collective_type = std.as_read(collective.value.expr_type)
  local value_type = collective_type.result_type

  local nparts = cx.graph:node_label(nparts_nid)

  local create_label = flow.node.Opaque {
    action = ast.typed.stat.Var {
      symbols = terralib.newlist({collective.value.value}),
      types = terralib.newlist({collective_type}),
      values = terralib.newlist({
          ast.typed.expr.DynamicCollective {
            arrivals = nparts.value,
            op = op,
            value_type = value_type,
            expr_type = collective_type,
            annotations = ast.default_annotations(),
            span = collective.value.span,
          },
      }),
      annotations = ast.default_annotations(),
      span = collective.value.span,
    }
  }
  local create_nid = cx.graph:add_node(create_label)

  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    nparts_nid, cx.graph:node_result_port(nparts_nid),
    create_nid, cx.graph:node_available_port(create_nid))

  cx.graph:add_edge(
    flow.edge.HappensBefore {}, create_nid, cx.graph:node_sync_port(create_nid),
    collective_nid, cx.graph:node_sync_port(collective_nid))
end

local function rewrite_local_collective_creation(cx, shard_loop, shard_vars,
                                                 local_collectives)
  for _, collective in ipairs(local_collectives) do
    local collective_label, op = unpack(collective)
    local collective_nid = find_matching_input(cx, shard_loop, collective_label.region_type, collective_label.field_path)

    local nparts_nid = find_matching_input(
      cx, shard_loop, shard_vars.nparts.region_type, shard_vars.nparts.field_path)

    issue_local_collective_creation(cx, shard_loop, collective_nid, op, nparts_nid)
  end

  -- Create a mapping to local collectives from propogating.
  local collective_mapping = {}
  for _, collective in ipairs(local_collectives) do
    local collective_label = unpack(collective)
    collective_mapping[collective_label.region_type] = false
  end
  return collective_mapping
end

local function rewrite_shard_intersections(cx, shard_loop, intersections)
  local mapping = {}
  local preconditions = {}

  local shallow_intersections = data.new_recursive_map(2)
  for lhs_type, i1 in intersections:items() do
    for rhs_type, i2 in i1:items() do
      local intersection_label
      for field_path, label in i2:items() do
        intersection_label = label
        break
      end
      assert(intersection_label)
      local intersection_type = intersection_label.region_type

      local shallow_type = std.list(
        std.list(intersection_type:subregion_dynamic(), nil, nil, nil, true),
      nil, nil, nil, true)
      local shallow_symbol = std.newsymbol(
        shallow_type, "shallow_" .. tostring(intersection_label.value.value))

      mapping[intersection_type] = shallow_type

      local lhs_label = cx.graph:node_label(find_matching_input(cx, shard_loop, lhs_type))

      local complete_label = flow.node.Opaque {
        action = ast.typed.stat.Var {
          symbols = terralib.newlist({intersection_label.value.value}),
          types = terralib.newlist({intersection_label.region_type}),
          values = terralib.newlist({
              ast.typed.expr.ListCrossProductComplete {
                lhs = lhs_label.value,
                product = intersection_label.value {
                  value = shallow_symbol,
                  expr_type = std.type_sub(intersection_label.value.expr_type, mapping),
                },
                expr_type = intersection_type,
                annotations = ast.default_annotations(),
                span = intersection_label.value.span,
              },
          }),
          annotations = ast.default_annotations(),
          span = intersection_label.value.span,
        }
      }
      local complete_nid = cx.graph:add_node(complete_label)

      preconditions[intersection_type] = complete_label

      for field_path, old_label in i2:items() do
        local old_nid = find_matching_input(cx, shard_loop, old_label.region_type, old_label.field_path)
        local lhs_nid = find_matching_input(cx, shard_loop, lhs_type, old_label.field_path)
        local new_label = old_label {
          value = old_label.value {
            value = shallow_symbol,
            expr_type = std.type_sub(old_label.value.expr_type, mapping),
          },
          region_type = shallow_type,
        }
        local new_nid = cx.graph:add_node(new_label)

        shallow_intersections[lhs_type][rhs_type][field_path] = new_label

        cx.graph:add_edge(
          flow.edge.None(flow.default_mode()), new_nid, cx.graph:node_result_port(new_nid),
          complete_nid, cx.graph:node_available_port(complete_nid))
        cx.graph:add_edge(
          flow.edge.None(flow.default_mode()), lhs_nid, cx.graph:node_result_port(lhs_nid),
          complete_nid, cx.graph:node_available_port(complete_nid))
        cx.graph:add_edge(
          flow.edge.HappensBefore {}, complete_nid, cx.graph:node_sync_port(complete_nid),
          old_nid, cx.graph:node_sync_port(old_nid))
      end
    end
  end

  return shallow_intersections, mapping, preconditions
end

-- If possible, we would prefer to initialize non-elided partitions inside of
-- shards instead of outside shards. However, this is only possible when the
-- elided partition dominates the non-elided one. Below we use a completeness
-- check on the elided partition for this purpose. But this check is only
-- valid when two types share the same parent, otherwise completeness is
-- irrelevant.
--
-- This function exists to make sure we only run complete when it makes sense.
local function permit_dominance_via_complete(cx, lhs_type, rhs_type)
  return std.type_eq(cx.tree:parent(lhs_type), cx.tree:parent(rhs_type))
end

local function rewrite_initial_copyin(cx, shard_loop, intersections, intersection_preconditions,
                                      elide_copy, elide_mapping)
  local inverse_mapping = {}
  for k, v in pairs(elide_mapping) do
    inverse_mapping[v] = k
  end

  local mappings = {}
  local preconditions = {}

  local shard_label = cx.graph:node_label(shard_loop)

  for lhs_type, i1 in intersections:items() do
    for rhs_type, i2 in i1:items() do
      if inverse_mapping[lhs_type] and inverse_mapping[rhs_type] and
        not elide_copy[inverse_mapping[rhs_type]] and
        permit_dominance_via_complete(
          cx, inverse_mapping[lhs_type], inverse_mapping[rhs_type])
      then
        local lhs_label = cx.graph:node_label(find_matching_input(cx, shard_loop, lhs_type))

        local intersection_label
        for field_path, label in i2:items() do
          intersection_label = label
          break
        end
        assert(intersection_label)
        local intersection_type = intersection_label.region_type
        local intersection_precondition_label =
          intersection_preconditions[intersection_type]
        local intersection_precondition_nid = cx.graph:find_node(
            function(_, label) return label == intersection_precondition_label end)

        local block_cx = cx:new_graph_scope(flow.empty_graph(cx.tree))
        local span = shard_label.span

        local cond_type = std.rawref(&int)
        local cond_symbol = std.newsymbol(int, "complete_cond")
        local cond_label = make_variable_label(block_cx, cond_symbol, int, span)
        local cond_nid = cx.graph:add_node(cond_label)
        mappings[cond_label.region_type] = false

        local check = flow.node.Opaque {
          action = ast.typed.stat.Var {
            symbols = terralib.newlist({cond_symbol}),
            types = terralib.newlist({int}),
            values = terralib.newlist({
                ast.typed.expr.Cast {
                  fn = ast.typed.expr.Function {
                    value = int,
                    expr_type = {std.untyped} -> int,
                    annotations = ast.default_annotations(),
                    span = span,
                  },
                  arg = ast.typed.expr.Call {
                    fn = ast.typed.expr.Function {
                      value = std.c.legion_index_partition_is_complete,
                      expr_type = std.c.legion_index_partition_is_complete:gettype(),
                      annotations = ast.default_annotations(),
                      span = span,
                    },
                    args = terralib.newlist({
                        ast.typed.expr.RawRuntime {
                          expr_type = std.c.legion_runtime_t,
                          annotations = ast.default_annotations(),
                          span = span,
                        },
                        ast.typed.expr.Call {
                          fn = ast.typed.expr.Function {
                            value = std.c.legion_index_space_get_parent_index_partition,
                            expr_type = std.c.legion_index_space_get_parent_index_partition:gettype(),
                            annotations = ast.default_annotations(),
                            span = span,
                          },
                          args = terralib.newlist({
                              ast.typed.expr.RawRuntime {
                                expr_type = std.c.legion_runtime_t,
                                annotations = ast.default_annotations(),
                                span = span,
                              },
                              ast.typed.expr.FieldAccess {
                                value = ast.typed.expr.RawValue {
                                  value = ast.typed.expr.IndexAccess {
                                    value = lhs_label.value,
                                    index = ast.typed.expr.Constant {
                                      value = 0,
                                      expr_type = int32,
                                      annotations = ast.default_annotations(),
                                      span = span,
                                    },
                                    expr_type = lhs_type:subregion_dynamic(),
                                    annotations = ast.default_annotations(),
                                    span = span,
                                  },
                                  expr_type = std.c.legion_logical_region_t,
                                  annotations = ast.default_annotations(),
                                  span = span,
                                },
                                field_name = "index_space",
                                expr_type = std.c.legion_index_space_t,
                                annotations = ast.default_annotations(),
                                span = span,
                              },
                          }),
                          conditions = terralib.newlist({}),
                          expr_type = std.c.legion_index_partition_t,
                          annotations = ast.default_annotations(),
                          span = span,
                        },
                    }),
                    conditions = terralib.newlist({}),
                    expr_type = bool,
                    annotations = ast.default_annotations(),
                    span = span,
                  },
                  expr_type = int,
                  annotations = ast.default_annotations(),
                  span = span,
                },
            }),
            annotations = ast.default_annotations(),
            span = span,
          }
        }
        local check_nid = cx.graph:add_node(check)
        cx.graph:add_edge(
          flow.edge.HappensBefore {}, intersection_precondition_nid,
          cx.graph:node_sync_port(intersection_precondition_nid),
          check_nid, cx.graph:node_sync_port(check_nid))
        cx.graph:add_edge(
          flow.edge.HappensBefore {}, check_nid, cx.graph:node_sync_port(check_nid),
          cond_nid, cx.graph:node_sync_port(cond_nid))

        local index_type = std.rawref(&int)
        local index_symbol = std.newsymbol(int, "complete_index")
        local index_label = make_variable_label(
          block_cx, index_symbol, int, span)
        local index_nid = block_cx.graph:add_node(index_label)

        local copy_loop = flow.node.ctrl.ForNum {
          symbol = index_symbol,
          block = block_cx.graph,
          annotations = ast.default_annotations(),
          span = span,
        }
        local copy_loop_nid = cx.graph:add_node(copy_loop)
        cx.graph:add_edge(
          flow.edge.HappensBefore {}, cond_nid, cx.graph:node_sync_port(cond_nid),
          copy_loop_nid, cx.graph:node_sync_port(copy_loop_nid))

        local bound1_nid = cx.graph:add_node(
          flow.node.Constant { value = make_constant(0, int, span) })
        local bound2_nid = cond_nid
        cx.graph:add_edge(
          flow.edge.Read(flow.default_mode()),
          bound1_nid, cx.graph:node_result_port(bound1_nid),
          copy_loop_nid, 1)
        cx.graph:add_edge(
          flow.edge.Read(flow.default_mode()),
          bound2_nid, cx.graph:node_result_port(bound2_nid),
          copy_loop_nid, 2)

        local inner_intersection_nid = block_cx.graph:add_node(intersection_label)

        local barrier_list_type = std.list(std.list(std.phase_barrier))
        local barrier_list_symbol =
          std.newsymbol(barrier_list_type, "copyin_barriers")
        local barrier_list_label =
          make_variable_label(block_cx, barrier_list_symbol, barrier_list_type, span)
        local barrier_list_nid = block_cx.graph:add_node(barrier_list_label)
        mappings[barrier_list_label.region_type] = false

        local get_barrier_list = flow.node.Opaque {
          action = ast.typed.stat.Var {
            symbols = terralib.newlist({barrier_list_symbol}),
            types = terralib.newlist({barrier_list_type}),
            values = terralib.newlist({
              ast.typed.expr.ListPhaseBarriers {
                product = intersection_label.value,
                expr_type = barrier_list_type,
                annotations = ast.default_annotations(),
                span = span,
              },
            }),
            annotations = ast.default_annotations(),
            span = span,
          }
        }

        local get_barrier_list_nid = block_cx.graph:add_node(get_barrier_list)
        block_cx.graph:add_edge(
          flow.edge.HappensBefore {},
          inner_intersection_nid, block_cx.graph:node_sync_port(inner_intersection_nid),
          get_barrier_list_nid, block_cx.graph:node_sync_port(get_barrier_list_nid))
        block_cx.graph:add_edge(
          flow.edge.HappensBefore {},
          get_barrier_list_nid, block_cx.graph:node_sync_port(get_barrier_list_nid),
          barrier_list_nid, block_cx.graph:node_sync_port(barrier_list_nid))

        local field_paths = terralib.newlist()
        for field_path, _ in i2:items() do field_paths:insert(field_path) end
        local copy = flow.node.Copy {
          src_field_paths = field_paths,
          dst_field_paths = field_paths,
          op = false,
          annotations = ast.default_annotations(),
          span = span,
        }
        local copy_nid = block_cx.graph:add_node(copy)

        block_cx.graph:add_edge(
          flow.edge.HappensBefore {},
          get_barrier_list_nid, block_cx.graph:node_sync_port(get_barrier_list_nid),
          copy_nid, block_cx.graph:node_sync_port(copy_nid))

        for field_path, old_label in i2:items() do
          local old_nid = find_matching_input(cx, shard_loop, old_label.region_type, old_label.field_path)
          local lhs_nid = find_matching_input(cx, shard_loop, lhs_type, old_label.field_path)
          local intersection_in_nid = cx.graph:add_node(old_label)
          local intersection_out_nid = cx.graph:add_node(old_label)

          cx.graph:add_edge(
            flow.edge.HappensBefore {}, cond_nid, cx.graph:node_sync_port(cond_nid),
            intersection_in_nid, cx.graph:node_sync_port(intersection_in_nid))
          cx.graph:add_edge(
            flow.edge.HappensBefore {},
            intersection_out_nid, cx.graph:node_sync_port(intersection_out_nid),
            old_nid, cx.graph:node_sync_port(old_nid))

          local copy_loop_src_port = cx.graph:node_available_port(copy_loop_nid)
          cx.graph:add_edge(
            flow.edge.Read(flow.default_mode()), lhs_nid, cx.graph:node_result_port(lhs_nid),
            copy_loop_nid, copy_loop_src_port)
          local copy_loop_dst_port = cx.graph:node_available_port(copy_loop_nid)
          cx.graph:add_edge(
            flow.edge.Read(flow.default_mode()), intersection_in_nid,
            cx.graph:node_result_port(intersection_in_nid),
            copy_loop_nid, copy_loop_dst_port)
          cx.graph:add_edge(
            flow.edge.Write(flow.default_mode()), copy_loop_nid, copy_loop_dst_port,
            intersection_out_nid, cx.graph:node_available_port(intersection_out_nid))

          local inner_lhs_nid = block_cx.graph:add_node(lhs_label)
          local inner_intersection_in_nid = block_cx.graph:add_node(old_label)
          local inner_intersection_out_nid = block_cx.graph:add_node(old_label)

          block_cx.graph:add_edge(
            flow.edge.Read(flow.default_mode()), inner_lhs_nid,
            block_cx.graph:node_result_port(inner_lhs_nid),
            copy_nid, 1)
          block_cx.graph:add_edge(
            flow.edge.Read(flow.default_mode()),
            inner_intersection_in_nid,
            block_cx.graph:node_result_port(inner_intersection_in_nid),
            copy_nid, 2)
          block_cx.graph:add_edge(
            flow.edge.Write(flow.default_mode()), copy_nid, 2,
            inner_intersection_out_nid,
            block_cx.graph:node_available_port(inner_intersection_out_nid))
        end

        block_cx.graph:add_edge(
          flow.edge.Arrive {},
          copy_nid, block_cx.graph:node_available_port(copy_nid),
          barrier_list_nid, block_cx.graph:node_available_port(barrier_list_nid))

        local sync_label = flow.node.Opaque {
          action = ast.typed.stat.Block {
            block = ast.typed.Block {
              stats = terralib.newlist({
                  ast.typed.stat.Assignment {
                    lhs = terralib.newlist({barrier_list_label.value}),
                    rhs = terralib.newlist({
                        ast.typed.expr.Advance {
                          value = barrier_list_label.value,
                          expr_type = barrier_list_type,
                          annotations = ast.default_annotations(),
                          span = shard_label.span,
                        }
                    }),
                    annotations = ast.default_annotations(),
                    span = shard_label.span,
                  },
                  ast.typed.stat.Expr {
                    expr = ast.typed.expr.Await {
                      barrier = barrier_list_label.value,
                      expr_type = terralib.types.unit,
                      annotations = ast.default_annotations(),
                      span = shard_label.span,
                    },
                    annotations = ast.default_annotations(),
                    span = shard_label.span,
                  },
              }),
              span = shard_label.span,
            },
            annotations = ast.default_annotations(),
            span = shard_label.span,
          }
        }
        local sync_nid = block_cx.graph:add_node(sync_label)

        block_cx.graph:add_edge(
          flow.edge.HappensBefore {}, copy_nid, block_cx.graph:node_sync_port(copy_nid),
          sync_nid, block_cx.graph:node_sync_port(sync_nid))

        preconditions[copy_loop] = true
      end
    end
  end

  return preconditions, mappings
end

-- Returns nids of IndexAccess nodes that access lists.
local function find_list_index_accesses(cx)
  return cx.graph:filter_nodes(
    function(nid, label)
      return label:is(flow.node.IndexAccess) and
        #cx.graph:filter_immediate_predecessors_by_edges(
          function(edge, label)
            return edge.to_port == 1 and label:is(flow.node.data) and
              std.is_list(std.as_read(label.value.expr_type))
          end,
          nid) > 0
    end)
end

-- Makes nodes that compute the global index in a loop body.
-- If `local_index_symbol == nil`, a new symbol will be created for the local index.
local function compute_global_index(block_cx, loop_label, global_index_symbol,
                                    local_index_symbol, shard_part_indices_label)
  if not local_index_symbol then
    local_index_symbol = std.newsymbol(loop_label.symbol.symbol_type, "local_index")
  end
  local local_index_label = make_variable_label(
    block_cx, local_index_symbol, local_index_symbol.symbol_type, loop_label.span)
  local local_index_nid = block_cx.graph:add_node(local_index_label)

  -- Find the old loop variable, which we'll keep as the global index.
  local global_index_nid = block_cx.graph:find_node(
    function(_, label)
      return label:is(flow.node.data.Scalar) and
        label.value.value == global_index_symbol
  end)
  local global_index_label = block_cx.graph:node_label(global_index_nid)
  local global_index_type = global_index_label.value.expr_type

  -- Compute global index (original loop variable) by indexing into the
  -- `shard_part_indices` list.
  local part_indices_nid = block_cx.graph:add_node(shard_part_indices_label)
  local compute_global_index_nid = block_cx.graph:add_node(flow.node.Opaque {
    action = ast.typed.stat.Var {  -- var global_index = shard_part_indices[local_index]
      symbols = terralib.newlist({global_index_symbol}),
      types = terralib.newlist({global_index_type}),
      values = terralib.newlist({
        ast.typed.expr.IndexAccess {
          value = shard_part_indices_label.value,
          index = local_index_label.value,
          expr_type = global_index_type,
          annotations = ast.default_annotations(),
          span = global_index_label.value.span
        },
      }),
      annotations = ast.default_annotations(),
      span = global_index_label.value.span,
    }
  })

  block_cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    part_indices_nid, block_cx.graph:node_result_port(part_indices_nid),
    compute_global_index_nid, 1)
  block_cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    local_index_nid, block_cx.graph:node_result_port(local_index_nid),
    compute_global_index_nid, 2)
  block_cx.graph:add_edge(
    flow.edge.HappensBefore {},
    compute_global_index_nid, block_cx.graph:node_sync_port(compute_global_index_nid),
    global_index_nid, block_cx.graph:node_sync_port(global_index_nid))

  return global_index_nid, local_index_nid
end

local function rewrite_inner_fornum_loop_bounds(cx, loop_label, shard_part_indices_label)
  local block_cx = cx:new_graph_scope(loop_label.block)
  local global_index_nid, local_index_nid = compute_global_index(
    block_cx, loop_label, loop_label.symbol, nil, shard_part_indices_label)

  -- Swap in the new loop variable.
  local local_index_symbol = block_cx.graph:node_label(local_index_nid).value.value
  loop_label.symbol = local_index_symbol

  -- Find any index accesses to lists and rewire them to use the local index.
  local index_access_nids = find_list_index_accesses(block_cx)
  for _, index_access_nid in ipairs(index_access_nids) do
    block_cx.graph:copy_outgoing_edges(
      function(edge)
        return block_cx.graph:reachable(
          edge.to_node, index_access_nid, isnt_happens_before)
      end,
    global_index_nid,
    local_index_nid,
    true)
  end
end

-- Propagates shard-specific variables to parent contexts of leaf loops.
--
-- Since the leaf loops may read from them, so may any node containing them.
-- This function updates `cached_nids` to contain nids for shard-specific
-- variables in each graph.
local function propagate_shard_vars(cx, shard_nparts_label,
                                    shard_part_indices_label, cached_nids)
  cx.graph:traverse_nodes_recursive(
    function(graph, nid, label)
      local inner_cx = cx:new_graph_scope(graph)
      if is_inner(inner_cx, nid) then
        if not cached_nids[graph] then
          cached_nids[graph] = {
            shard_nparts = inner_cx.graph:add_node(shard_nparts_label),
            shard_part_indices = inner_cx.graph:add_node(shard_part_indices_label),
          }
        end

        local shard_nparts_nid = cached_nids[graph].shard_nparts
        inner_cx.graph:add_edge(
          flow.edge.Read(flow.default_mode()),
          shard_nparts_nid, inner_cx.graph:node_result_port(shard_nparts_nid),
          nid, inner_cx.graph:node_available_port(nid))

        local shard_part_indices_nid = cached_nids[graph].shard_part_indices
        inner_cx.graph:add_edge(
          flow.edge.Read(flow.default_mode()),
          shard_part_indices_nid, inner_cx.graph:node_result_port(shard_part_indices_nid),
          nid, inner_cx.graph:node_available_port(nid))
      end
    end)
end

local function rewrite_fornum_loops_in_shard(cx, shard_loop)
  local shard_label = cx.graph:node_label(shard_loop)

  -- Find the current loop bounds.
  local original_bounds_labels
  local bounds_type
  cx.graph:traverse_nodes_recursive(
    function(graph, nid, label)
      local inner_cx = cx:new_graph_scope(graph)
      if is_leaf(inner_cx, nid) then
        assert(label:is(flow.node.ctrl.ForNum))
        local inputs = inner_cx.graph:incoming_edges_by_port(nid)
        local value1 = inner_cx.graph:node_label(get_input(inputs, 1))
        local value2 = inner_cx.graph:node_label(get_input(inputs, 2))
        if not original_bounds_labels then
          original_bounds_labels = terralib.newlist({value1, value2})
        end

        local value_type = std.type_meet(
          std.as_read(value1.value.expr_type),
          std.as_read(value2.value.expr_type))
        if bounds_type then
          assert(std.type_eq(value_type, bounds_type))
        end
        bounds_type = value_type
      end
    end)
  assert(original_bounds_labels and bounds_type)

  -- Make labels for shard-specific variables.
  local shard_nparts_label = make_variable_label(
    cx, std.newsymbol(bounds_type, "shard_nparts"), bounds_type, shard_label.span)
  local shard_part_indices_label = make_variable_label(
    cx, std.newsymbol(std.list(bounds_type), "shard_part_indices"),
    std.list(bounds_type), shard_label.span)

  -- Replace old bounds with new.
  local cached_nids = {}
  cx.graph:traverse_nodes_recursive(
    function(graph, nid, label)
      local inner_cx = cx:new_graph_scope(graph)
      if is_leaf(inner_cx, nid) then
        if not cached_nids[graph] then
          cached_nids[graph] = {
            shard_nparts = inner_cx.graph:add_node(shard_nparts_label),
            shard_part_indices = inner_cx.graph:add_node(shard_part_indices_label),
          }
        end

        local new_bounds = terralib.newlist({
          inner_cx.graph:add_node(flow.node.Constant {
            value = make_constant(0, bounds_type, shard_nparts_label.value.span),
          }), -- Lower bound (constant 0).
          cached_nids[graph].shard_nparts -- Upper bound.
        })
        -- Change bounds.
        local inputs = inner_cx.graph:incoming_edges_by_port(nid)
        for i = 1, 2 do
          local value_nid, edge = get_input(inputs, i)
          local value_inputs = inner_cx.graph:incoming_edges(value_nid)
          for _, edge in ipairs(value_inputs) do
            assert(is_happens_before(edge))
          end
          inner_cx.graph:replace_edges(edge.to_node, edge.from_node, new_bounds[i])
          if #inner_cx.graph:outgoing_edges(value_nid) == 0 then
            inner_cx.graph:remove_node(value_nid)
          end
        end
        -- At this point, the loop variable has a different meaning: it is now
        -- the local index rather than the global index.  The original loop
        -- variable is kept around, and its value will be computed from the
        -- local index.

        -- Loop may also read from `shard_part_indices` (when the loop body
        -- uses the global index).
        local shard_part_indices_nid = cached_nids[graph].shard_part_indices
        inner_cx.graph:add_edge(
          flow.edge.Read(flow.default_mode()),
          shard_part_indices_nid, inner_cx.graph:node_result_port(shard_part_indices_nid),
          nid, inner_cx.graph:node_available_port(nid))
      end
    end)

  -- Rewrite bounds used for accessing lists inside of loops.
  cx.graph:traverse_nodes_recursive(
    function(graph, nid, label)
      local inner_cx = cx:new_graph_scope(graph)
      if is_leaf(inner_cx, nid) then
        rewrite_inner_fornum_loop_bounds(inner_cx, label, shard_part_indices_label)
      end
    end)

  propagate_shard_vars(cx, shard_nparts_label, shard_part_indices_label, cached_nids)

  local shard_vars = { nparts = shard_nparts_label, part_indices = shard_part_indices_label }
  return shard_vars, original_bounds_labels
end

-- Replaces a leaf ForList node with a ForNum.
-- Note: this function does NOT rewrite the loop body.
local function replace_forlist_node(cx, forlist_nid,
                                    shard_nparts_nid, shard_part_indices_nid)

  local forlist_label = cx.graph:node_label(forlist_nid)
  local global_index = forlist_label.symbol

  local local_index = std.newsymbol(int, "local_index")
  local fornum_label = flow.node.ctrl.ForNum {
    symbol = local_index,
    -- The loop body will need to be rewritten with `local_index` as loop variable.
    block = forlist_label.block,
    annotations = forlist_label.annotations,
    span = forlist_label.span,
  }
  local fornum_nid = cx.graph:add_node(fornum_label)

  -- Connect ForNum loop bounds.
  local lb_nid = cx.graph:add_node(flow.node.Constant {
    value = make_constant(0, int, forlist_label.span)
  })
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    lb_nid, cx.graph:node_result_port(lb_nid), fornum_nid, 1)
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    shard_nparts_nid, cx.graph:node_result_port(shard_nparts_nid), fornum_nid, 2)

  -- Move over edges from the old ForList node.
  -- We add 2 to each edge's port number (on the ForNum side) since the minimum
  -- available port is 2 for ForList but is 4 for ForNum.
  cx.graph:traverse_incoming_edges(
    function(from_node, to_node, edge)
      if edge.to_port ~= 1 then  -- Omit the edge wiring the old loop bound.
        cx.graph:add_edge(edge.label, from_node, edge.from_port,
          fornum_nid, edge.to_port + 2)
      end
      cx.graph:remove_edge(from_node, edge.from_port, to_node, edge.to_port)
    end,
    forlist_nid)
  cx.graph:traverse_outgoing_edges(
    function(from_node, to_node, edge)
      cx.graph:add_edge(edge.label, fornum_nid, edge.from_port + 2,
        to_node, edge.to_port)
      cx.graph:remove_edge(from_node, edge.from_port, to_node, edge.to_port)
    end,
    forlist_nid)
  -- The loop body might also read from `shard_part_indices` (when it uses the
  -- old ForList loop variable).
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    shard_part_indices_nid, cx.graph:node_result_port(shard_part_indices_nid),
    fornum_nid, cx.graph:node_available_port(fornum_nid))

  -- At this point, the original ForList node should have no edges coming into
  -- or out of it.  We delete it.
  assert(#cx.graph:incoming_edges(forlist_nid) == 0)
  assert(#cx.graph:outgoing_edges(forlist_nid) == 0)
  cx.graph:remove_node(forlist_nid)

  return fornum_nid, fornum_label, global_index
end

-- Rewrites the body of an old ForList loop after it's put into a ForNum loop.
local function rewrite_forlist_loop_body(cx, new_loop_label,
                                         shard_part_indices_label,
                                         global_index_symbol)

  assert(new_loop_label:is(flow.node.ctrl.ForNum))
  local block_cx = cx:new_graph_scope(new_loop_label.block)

  local _, local_index_nid = compute_global_index(
    block_cx, new_loop_label, global_index_symbol, new_loop_label.symbol,
    shard_part_indices_label)

  -- Find index accesses to lists and rewire them to use local index.
  local index_access_nids = find_list_index_accesses(block_cx)
  for _, index_access_nid in ipairs(index_access_nids) do
    block_cx.graph:remove_incoming_edges(
      function(edge) return edge.to_port == 2 end,
      index_access_nid)
    block_cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()),
      local_index_nid, block_cx.graph:node_result_port(local_index_nid),
      index_access_nid, 2)
  end
end

local function rewrite_forlist_loops_in_shard(cx, shard_loop)
  local shard_label = cx.graph:node_label(shard_loop)

  -- Find the current loop bounds, and collect all ForList loops along the way.
  local original_bound_label
  local forlist_loops = terralib.newlist()
  cx.graph:traverse_nodes_recursive(
    function(graph, nid, label)
      local inner_cx = cx:new_graph_scope(graph)
      if is_leaf(inner_cx, nid) then
        assert(label:is(flow.node.ctrl.ForList))
        local inputs = inner_cx.graph:incoming_edges_by_port(nid)
        original_bound_label = inner_cx.graph:node_label(get_input(inputs, 1))
        forlist_loops:insert({ cx = inner_cx, nid = nid })
      end
    end)
  assert(original_bound_label)
  local bound_type = std.as_read(original_bound_label.value.expr_type)
  local index_type = bound_type.index_type
  assert(index_type)

  -- Make labels for shard-specific variables.
  local shard_nparts_label = make_variable_label(
    cx, std.newsymbol(int, "shard_nparts"), int, shard_label.span)
  local shard_part_indices_label = make_variable_label(
    cx, std.newsymbol(std.list(index_type), "shard_part_indices"),
    std.list(index_type), shard_label.span)

  -- Replace each ForList loop with a ForNum loop from 0 to `shard_nparts`.
  local cached_nids = {}
  for _, forlist_loop in ipairs(forlist_loops) do
    local cx, forlist_nid = forlist_loop.cx, forlist_loop.nid
    if not cached_nids[cx.graph] then
      cached_nids[cx.graph] = {
        shard_nparts = cx.graph:add_node(shard_nparts_label),
        shard_part_indices = cx.graph:add_node(shard_part_indices_label),
      }
    end

    local new_loop_nid, new_loop_label, global_index = replace_forlist_node(cx,
      forlist_nid, cached_nids[cx.graph].shard_nparts,
      cached_nids[cx.graph].shard_part_indices)
    rewrite_forlist_loop_body(cx, new_loop_label, shard_part_indices_label, global_index)
  end

  propagate_shard_vars(cx, shard_nparts_label, shard_part_indices_label, cached_nids)

  local shard_vars = { nparts = shard_nparts_label, part_indices = shard_part_indices_label }
  return shard_vars, terralib.newlist({original_bound_label})
end

-- Makes labels for variables pertaining to the entire SPMD block.
local function make_global_vars(cx, shard_vars, span)
  local bounds_type = std.as_read(shard_vars.nparts.value.expr_type)
  local nparts_label = make_variable_label(
    cx, std.newsymbol(bounds_type, "nparts"), bounds_type, span)

  local part_indices_type = std.as_read(shard_vars.part_indices.value.expr_type)
  local part_indices_label = make_variable_label(
    cx, std.newsymbol(part_indices_type, "part_indices"), part_indices_type, span)

  return { nparts = nparts_label, part_indices = part_indices_label }
end

local function synchronize_shard_start(cx, shard_loop, precondition_labels)
  local shard_label = cx.graph:node_label(shard_loop)

  local precondition_nids = cx.graph:filter_nodes(
    function(_, label) return precondition_labels[label] ~= nil end)

  local barrier_type = std.phase_barrier
  local barrier_symbol = std.newsymbol(barrier_type, "start_barrier")
  local barrier_label = make_variable_label(cx, barrier_symbol, barrier_type, shard_label.span)
  local barrier_nid = cx.graph:add_node(barrier_label)

  local sync_label = flow.node.Opaque {
    action = ast.typed.stat.Block {
      block = ast.typed.Block {
        stats = terralib.newlist({
            ast.typed.stat.Expr {
              expr = ast.typed.expr.Arrive {
                barrier = barrier_label.value,
                value = false,
                expr_type = barrier_type,
                annotations = ast.default_annotations(),
                span = shard_label.span,
              },
              annotations = ast.default_annotations(),
              span = shard_label.span,
            },
            ast.typed.stat.Assignment {
              lhs = terralib.newlist({barrier_label.value}),
              rhs = terralib.newlist({
                  ast.typed.expr.Advance {
                    value = barrier_label.value,
                    expr_type = barrier_type,
                    annotations = ast.default_annotations(),
                    span = shard_label.span,
                  }
              }),
              annotations = ast.default_annotations(),
              span = shard_label.span,
            },
            ast.typed.stat.Expr {
              expr = ast.typed.expr.Await {
                barrier = barrier_label.value,
                expr_type = terralib.types.unit,
                annotations = ast.default_annotations(),
                span = shard_label.span,
              },
              annotations = ast.default_annotations(),
              span = shard_label.span,
            },
        }),
        span = shard_label.span,
      },
      annotations = ast.default_annotations(),
      span = shard_label.span,
    }
  }
  local sync_nid = cx.graph:add_node(sync_label)

  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    barrier_nid, cx.graph:node_result_port(barrier_nid),
    sync_nid, cx.graph:node_available_port(sync_nid))
  for _, nid in ipairs(precondition_nids) do
    cx.graph:add_edge(
      flow.edge.HappensBefore {},
      nid, cx.graph:node_sync_port(nid),
      sync_nid, cx.graph:node_sync_port(sync_nid))
  end
  cx.graph:add_edge(
    flow.edge.HappensBefore {},
    sync_nid, cx.graph:node_sync_port(sync_nid),
    shard_loop, cx.graph:node_sync_port(shard_loop))

  return barrier_label
end

local function select_copy_elision(cx, lists, mapping)
  local inverse_mapping = {}
  for k, v in pairs(mapping) do
    inverse_mapping[v] = k
  end

  -- In general, partitions must be duplicated to avoid conflicts. But
  -- disjoint partitions (at most one, if there are multiple such
  -- aliased partitions) need not be duplicated---and in avoiding the
  -- duplication, copies can also be avoided.

  local elide_copy = data.newmap()
  local elide_none = data.newmap()
  for _, list_type in lists:keys() do
    if std.is_list_of_regions(list_type) then
      local list_nids = cx.graph:filter_nodes(
        function(nid, label)
          return label:is(flow.node.data) and std.type_eq(label.region_type, list_type) end)
      local has_privilege = data.any(
        unpack(
          list_nids:map(
            function(nid)
              return #cx.graph:incoming_mutate_set(nid) > 0 or #cx.graph:outgoing_read_set(nid) > 0
            end)))

      local old_type = inverse_mapping[list_type]
      if not has_privilege then
        -- Keep these in a separate list for now. Don't want to
        -- accidentally shadow something.
        elide_none[old_type] = true
      elseif std.is_partition(old_type) and old_type:is_disjoint() then
        local aliased = false
        for other_type, _ in elide_copy:items() do
          if std.type_maybe_eq(old_type:fspace(), other_type:fspace()) and
            cx.tree:can_alias(old_type, other_type)
          then
            aliased = true
            break
          end
        end
        if not aliased then
          elide_copy[old_type] = true
        end
      end
    end
  end

  -- Build a map of who shadows who based on which copies were elided.
  local shadowed_partitions = data.newmap()
  for _, list_type in lists:keys() do
    local old_type = inverse_mapping[list_type]
    if std.is_list_of_regions(list_type) and std.is_partition(old_type) then
      for other_type, _ in elide_copy:items() do
        if not std.type_eq(old_type, other_type) and
          std.type_maybe_eq(old_type:fspace(), other_type:fspace()) and
          cx.tree:can_alias(old_type, other_type)
        then
          shadowed_partitions[old_type] = other_type
          break
        end
      end
    end
  end

  -- Apply none-privilege list to elided copies.
  for old_type, _ in elide_none:items() do
    elide_copy[old_type] = true
  end

  return elide_copy, shadowed_partitions
end

local function rewrite_elided_lists(cx, lists, intersections, barriers,
                                    inverse_mapping, elide_copy)
  local elided_lists = data.new_recursive_map(1)
  local elided_mapping = {}
  for list_type, list in lists:items() do
    local original_type = inverse_mapping[list_type]
    if elide_copy[original_type] then
      local _, list_label = cx.graph:find_node(
        function(_, label) return label:is(flow.node.data) and label.region_type == list_type end)

      local new_type = std.list(list_type.element_type, list_type.partition_type, nil, original_type:parent_region())
      local new_symbol = std.newsymbol(new_type, "elided_" .. tostring(list_label.value.value))
      cx.tree:intern_region_expr(
        new_type, ast.default_annotations(), list_label.value.span)

      elided_mapping[list_type] = new_type

      -- Rewrite the list entries.
      for field_path, label in list:items() do
        elided_lists[new_type][field_path] = label {
          value = label.value {
            value = new_symbol,
            expr_type = std.type_sub(label.value.expr_type, elided_mapping),
          },
          region_type = new_type,
        }
      end

      -- Rewrite the graph.
      cx.graph:map_nodes_recursive(
        function(_, _, label)
          if label:is(flow.node.data) and label.region_type == list_type then
            return label {
              value = label.value {
                value = new_symbol,
                expr_type = std.type_sub(label.value.expr_type, elided_mapping),
              },
              region_type = new_type,
            }
          end
          return label
        end)
    else
      elided_lists[list_type] = list
    end
  end

  local elided_intersections = data.new_recursive_map(2)
  for lhs_type, i1 in intersections:items() do
    for rhs_type, i2 in i1:items() do
      local new_lhs_type = std.type_sub(lhs_type, elided_mapping)
      local new_rhs_type = std.type_sub(rhs_type, elided_mapping)

      if elide_copy[inverse_mapping[rhs_type]] then
        local old_label
        for field_path, label in i2:items() do
          old_label = label
        end
        assert(old_label)

        local old_type = old_label.region_type
        local new_type = std.list(
          std.list(old_type.element_type.element_type, old_type.partition_type,
                   new_rhs_type.privilege_depth, new_rhs_type.region_root,
                   old_type.shallow),
          old_type.partition_type,
          new_rhs_type.privilege_depth, new_rhs_type.region_root,
          old_type.shallow)
        local new_symbol = std.newsymbol(new_type, "elided_" .. tostring(old_label.value.value))
        cx.tree:intern_region_expr(
          new_type, ast.default_annotations(), old_label.value.span)

        elided_mapping[old_type] = new_type

        -- Rewrite the intersection entries.
        for field_path, label in i2:items() do
          elided_intersections[new_lhs_type][new_rhs_type][field_path] = label {
            value = label.value {
              value = new_symbol,
              expr_type = std.type_sub(label.value.expr_type, elided_mapping),
            },
            region_type = new_type,
          }
        end

      -- Rewrite the graph.
      cx.graph:map_nodes_recursive(
        function(_, _, label)
          if label:is(flow.node.data) and label.region_type == old_type then
            return label {
              value = label.value {
                value = new_symbol,
                expr_type = std.type_sub(label.value.expr_type, elided_mapping),
              },
              region_type = new_type,
            }
          end
          return label
        end)
      else
        for field_path, label in i2:items() do
          elided_intersections[new_lhs_type][new_rhs_type][field_path] = label
        end
      end
    end
  end

  local elided_barriers = data.new_recursive_map(2)
  for lhs_type, b1 in barriers:items() do
    for rhs_type, b2 in b1:items() do
      for field_path, barrier_labels in b2:items() do
        local new_lhs_type = std.type_sub(lhs_type, elided_mapping)
        local new_rhs_type = std.type_sub(rhs_type, elided_mapping)
        elided_barriers[new_lhs_type][new_rhs_type][field_path] = barrier_labels
      end
    end
  end

  return elided_lists, elided_intersections, elided_barriers, elided_mapping
end

local function rewrite_list_elision(cx, lists, intersections, barriers, mapping,
                                    elide_copy, shadow_partitions)
  -- Rewrite the elided lists.
  local inverse_mapping = {}
  for k, v in pairs(mapping) do
    inverse_mapping[v] = k
  end

  local elided_lists, elided_intersections, elided_barriers, elided_mapping =
    rewrite_elided_lists(cx, lists, intersections, barriers,
                         inverse_mapping, elide_copy)
  return elided_lists, elided_intersections, elided_barriers, elided_mapping
end

local function get_slice_type_and_symbol(cx, region_type, list_type, label)
  if std.is_list_of_regions(region_type) or
    std.is_list_of_partitions(region_type)
  then
    local parent_list_type = list_type:slice()
    for other_region, _ in pairs(cx.tree.region_universe) do
      assert(not std.type_eq(parent_list_type, other_region))
      if not std.type_eq(other_region, list_type) and
        std.type_maybe_eq(parent_list_type:fspace(), other_region:fspace())
      then
        std.add_constraint(cx.tree, parent_list_type, other_region, std.disjointness, true)
      end
    end
    cx.tree:intern_region_expr(
      parent_list_type, ast.default_annotations(), label.value.span)

    local parent_list_symbol = std.newsymbol(
      parent_list_type,
      "parent_" .. tostring(label.value.value))
    return parent_list_type, parent_list_type, parent_list_symbol
  else
    local parent_list_type = std.rawref(&list_type)
    local parent_list_symbol = std.newsymbol(
      list_type,
      "parent_" .. tostring(label.value.value))
    local parent_list_region = cx.tree:intern_variable(
      parent_list_type, parent_list_symbol,
      ast.default_annotations(), label.value.span)
    return parent_list_type, parent_list_region, parent_list_symbol
  end
end

local function build_slice(cx, region_type, list_type,
                           shard_list_indices_nid, shard_list_indices_label,
                           bounds_type, slice_mapping)
  local list_nids = cx.graph:filter_nodes(
    function(nid, label)
      return label:is(flow.node.data) and
        label.region_type == region_type and
        #cx.graph:immediate_predecessors(nid) == 0
    end)
  if #list_nids > 0 then
    -- Grab one of them so we can make the slice...
    local first_list = cx.graph:node_label(list_nids[1])

    local parent_list_type, parent_list_region, parent_list_symbol =
      get_slice_type_and_symbol(cx, region_type, list_type, first_list)
    slice_mapping[region_type] = parent_list_region
    local first_parent_list = first_list {
      region_type = parent_list_region,
      value = first_list.value {
        value = parent_list_symbol,
        expr_type = std.type_sub(first_list.value.expr_type, slice_mapping),
      }
    }

    local compute_list = flow.node.Opaque {
      action = ast.typed.expr.IndexAccess {
        value = first_parent_list.value,
        index = shard_list_indices_label.value,
        expr_type = std.as_read(first_list.value.expr_type),
        annotations = ast.default_annotations(),
        span = first_list.value.span,
      },
    }
    local compute_list_nid = cx.graph:add_node(compute_list)

    for _, list_nid in ipairs(list_nids) do
      local list = cx.graph:node_label(list_nid)
      local parent_list = list {
        region_type = parent_list_region,
        value = list.value {
          value = parent_list_symbol,
          expr_type = std.type_sub(list.value.expr_type, slice_mapping),
        },
      }
      local parent_nid = cx.graph:add_node(parent_list)

      cx.graph:add_edge(
        flow.edge.Name {},
        compute_list_nid, cx.graph:node_result_port(compute_list_nid),
        list_nid, 1)
      cx.graph:add_edge(
        flow.edge.None(flow.default_mode()), parent_nid, cx.graph:node_result_port(parent_nid),
        compute_list_nid, cx.graph:node_available_port(compute_list_nid))
    end
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()),
      shard_list_indices_nid, cx.graph:node_result_port(shard_list_indices_nid),
      compute_list_nid, cx.graph:node_available_port(compute_list_nid))
    return first_parent_list
  end
end

local function rewrite_shard_slices(cx, shard_vars, global_vars, lists,
                                    intersections, barriers, shard_task_nid, mapping)
  local slice_mapping = {}

  -- Build the actual shard index.
  local bounds_type = std.as_read(shard_vars.nparts.value.expr_type)
  local index_label = make_variable_label(
    cx, std.newsymbol(bounds_type, "shard_index"),
    bounds_type, shard_vars.nparts.value.span)
  local index_nid = cx.graph:add_node(index_label)

  -- Build shard stride (i.e. size of each shard). Currently constant.
  local stride_label = flow.node.Constant {
    value = make_constant(shard_size, bounds_type, index_label.value.span),
  }
  local stride_nid = cx.graph:add_node(stride_label)

  -- Make labels for shard bounds.  These don't get directly passed into the
  -- shard task; they're temporaries used to compute the shard_vars, which get
  -- passed into the shard task.
  local bound_labels = terralib.newlist()
  for i = 1, 2 do
    local bound_label = make_variable_label(
      cx, std.newsymbol(bounds_type, "shard_bound" .. i),
      bounds_type, index_label.value.span)
    bound_labels:insert(bound_label)
  end
  local bound_nids = bound_labels:map(
    function(bound_label)
      return cx.graph:add_node(bound_label)
    end)

  -- Use index and stride to compute shard bounds.
  local nparts_nid = cx.graph:add_node(global_vars.nparts)
  local compute_bounds = flow.node.Opaque {
    action = ast.typed.stat.Var {
      symbols = bound_labels:map(function(bound) return bound.value.value end),
      types = bound_labels:map(
        function(bound) return std.as_read(bound.value.expr_type) end),
      values = terralib.newlist({
          index_label.value,
          ast.typed.expr.Binary {
            lhs = ast.typed.expr.Binary {
              lhs = index_label.value,
              rhs = stride_label.value,
              op = "+",
              expr_type = std.as_read(index_label.value.expr_type),
              annotations = ast.default_annotations(),
              span = index_label.value.span,
            },
            rhs = global_vars.nparts.value,
            op = "min",
            expr_type = std.as_read(index_label.value.expr_type),
            annotations = ast.default_annotations(),
            span = index_label.value.span,
          },
      }),
      annotations = ast.default_annotations(),
      span = index_label.value.span,
    }
  }
  local compute_bounds_nid = cx.graph:add_node(compute_bounds)
  for _, bound_nid in ipairs(bound_nids) do
    cx.graph:add_edge(
      flow.edge.HappensBefore {},
      compute_bounds_nid, cx.graph:node_sync_port(compute_bounds_nid),
      bound_nid, cx.graph:node_sync_port(bound_nid))
  end
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()), index_nid, cx.graph:node_result_port(index_nid),
    compute_bounds_nid, cx.graph:node_available_port(compute_bounds_nid))
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()), stride_nid, cx.graph:node_result_port(stride_nid),
    compute_bounds_nid, cx.graph:node_available_port(compute_bounds_nid))
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()), nparts_nid, cx.graph:node_result_port(nparts_nid),
    compute_bounds_nid, cx.graph:node_available_port(nparts_nid))

  -- Compute `shard_nparts = shard_bound2 - shard_bound1`.
  local shard_nparts_nid = find_matching_input(cx, shard_task_nid,
    shard_vars.nparts.region_type, shard_vars.nparts.field_path)
  -- A node for `shard_nparts` should have been added to this context as a
  -- predecessor to the shard task node because the task takes it as input.
  assert(shard_nparts_nid)
  local compute_shard_nparts_nid = cx.graph:add_node(flow.node.Opaque {
    action = ast.typed.stat.Var {
      symbols = terralib.newlist({shard_vars.nparts.value.value}),
      types = terralib.newlist({bounds_type}),
      values = terralib.newlist({
        ast.typed.expr.Binary {
          lhs = bound_labels[2].value,
          rhs = bound_labels[1].value,
          op = "-",
          expr_type = bounds_type,
          annotations = ast.default_annotations(),
          span = index_label.value.span,
        },
      }),
      annotations = ast.default_annotations(),
      span = index_label.value.span,
    }
  })
  for i, bound_nid in ipairs(bound_nids) do
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()),
      bound_nid, cx.graph:node_result_port(compute_bounds_nid),
      compute_shard_nparts_nid, i)
  end
  cx.graph:add_edge(
    flow.edge.HappensBefore {},
    compute_shard_nparts_nid, cx.graph:node_sync_port(compute_shard_nparts_nid),
    shard_nparts_nid, cx.graph:node_sync_port(shard_nparts_nid))

  -- Make a list of 1d indices (shard_bound1 .. shard_bound2) for shard.
  -- It's used to slice the various lists (e.g. of barriers) to get the
  -- portions for this shard we're launching.
  local shard_list_indices_type = std.list(bounds_type)
  local shard_list_indices_label = make_variable_label(
    cx, std.newsymbol(shard_list_indices_type, "shard_list_indices"),
    shard_list_indices_type, shard_vars.nparts.value.span)
  local shard_list_indices_nid = cx.graph:add_node(shard_list_indices_label)
  -- `var shard_list_indices = list_range(shard_bound1, shard_bound2)`.
  local compute_shard_list_indices_nid = cx.graph:add_node(flow.node.Opaque {
    action = ast.typed.stat.Var {
      symbols = terralib.newlist({shard_list_indices_label.value.value}),
      types = terralib.newlist({shard_list_indices_type}),
      values = terralib.newlist({
        ast.typed.expr.ListRange {
          start = bound_labels[1].value,
          stop = bound_labels[2].value,
          expr_type = shard_list_indices_type,
          annotations = ast.default_annotations(),
          span = index_label.value.span,
        }
      }),
      annotations = ast.default_annotations(),
      span = index_label.value.span,
    }
  })
  for _, bound_nid in ipairs(bound_nids) do
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()), bound_nid, cx.graph:node_result_port(bound_nid),
      compute_shard_list_indices_nid, cx.graph:node_available_port(compute_shard_list_indices_nid))
  end
  cx.graph:add_edge(
    flow.edge.HappensBefore {},
    compute_shard_list_indices_nid, cx.graph:node_sync_port(compute_shard_list_indices_nid),
    shard_list_indices_nid, cx.graph:node_sync_port(shard_list_indices_nid))

  -- Slice the global `part_indices` to make list of partition indices for shard.
  local shard_part_indices_label = shard_vars.part_indices
  local shard_part_indices_nid = find_matching_input(cx, shard_task_nid,
    shard_part_indices_label.region_type, shard_part_indices_label.field_path)
  assert(shard_part_indices_nid)
  local shard_part_indices_type = std.as_read(shard_part_indices_label.value.expr_type)
  -- `var shard_part_indices = part_indices[shard_list_indices]`.
  local compute_shard_part_indices_nid = cx.graph:add_node(flow.node.Opaque {
    action = ast.typed.stat.Var {
      symbols = terralib.newlist({shard_part_indices_label.value.value}),
      types = terralib.newlist({shard_part_indices_type}),
      values = terralib.newlist({
        ast.typed.expr.IndexAccess {
          value = global_vars.part_indices.value,
          index = shard_list_indices_label.value,
          expr_type = shard_part_indices_type,
          annotations = ast.default_annotations(),
          span = index_label.value.span,
        },
      }),
      annotations = ast.default_annotations(),
      span = index_label.value.span,
    }
  })
  local part_indices_nid = cx.graph:add_node(global_vars.part_indices)
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    part_indices_nid, cx.graph:node_result_port(part_indices_nid),
    compute_shard_part_indices_nid,
    cx.graph:node_available_port(compute_shard_part_indices_nid))
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    shard_list_indices_nid, cx.graph:node_result_port(shard_list_indices_nid),
    compute_shard_part_indices_nid,
    cx.graph:node_available_port(compute_shard_part_indices_nid))
  cx.graph:add_edge(
    flow.edge.HappensBefore {},
    compute_shard_part_indices_nid, cx.graph:node_sync_port(compute_shard_part_indices_nid),
    shard_part_indices_nid, cx.graph:node_sync_port(shard_part_indices_nid))

  -- Kill local bounds in the slice mapping.
  slice_mapping[shard_vars.nparts.region_type] = false
  slice_mapping[shard_vars.part_indices.region_type] = false
  slice_mapping[index_label.region_type] = false
  slice_mapping[shard_list_indices_label.region_type] = false
  bound_labels:map(
    function(bound_label)
      slice_mapping[bound_label.region_type] = false
    end)

  -- Build list slices for original lists.
  for _, list_type in lists:keys() do
    build_slice(cx, list_type, list_type,
      shard_list_indices_nid, shard_list_indices_label,
      bounds_type, slice_mapping)
  end

  -- Build list slices for intersections.
  local parent_intersections = data.new_recursive_map(2)
  for lhs_type, i1 in intersections:items() do
    for rhs_type, i2 in i1:items() do
      local list_type
      for field_path, intersection_label in i2:items() do
        list_type = intersection_label.region_type
        break
      end
      assert(list_type)

      local parent = build_slice(cx, list_type, list_type,
        shard_list_indices_nid, shard_list_indices_label,
        bounds_type, slice_mapping)
      assert(parent)

      for field_path, intersection_label in i2:items() do
        parent_intersections[slice_mapping[lhs_type]][slice_mapping[rhs_type]][field_path] =
          parent
      end
    end
  end

  local parent_barriers = data.new_recursive_map(2)
  for lhs_type, b1 in barriers:items() do
    for rhs_type, b2 in b1:items() do
      for field_path, barrier_labels in b2:items() do
        local empty_in, empty_out, full_in, full_out = unpack(barrier_labels)

        local empty_in_type = std.as_read(empty_in.value.expr_type)
        local empty_in_region = empty_in.region_type
        local empty_in_parent = build_slice(
          cx, empty_in_region, empty_in_type,
          shard_list_indices_nid, shard_list_indices_label,
          bounds_type, slice_mapping)

        local empty_out_type = std.as_read(empty_out.value.expr_type)
        local empty_out_region = empty_out.region_type
        local empty_out_parent = build_slice(
          cx, empty_out_region, empty_out_type,
          shard_list_indices_nid, shard_list_indices_label,
          bounds_type, slice_mapping)

        local full_in_type = std.as_read(full_in.value.expr_type)
        local full_in_region = full_in.region_type
        local full_in_parent = build_slice(
          cx, full_in_region, full_in_type,
          shard_list_indices_nid, shard_list_indices_label,
          bounds_type, slice_mapping)

        local full_out_type = std.as_read(full_out.value.expr_type)
        local full_out_region = full_out.region_type
        local full_out_parent = build_slice(
          cx, full_out_region, full_out_type,
          shard_list_indices_nid, shard_list_indices_label,
          bounds_type, slice_mapping)

        assert(empty_in_parent and empty_out_parent and
                 full_in_parent and full_out_parent)
        parent_barriers[slice_mapping[lhs_type]][slice_mapping[rhs_type]][
          field_path] = data.newtuple(empty_in_parent, empty_out_parent,
                                      full_in_parent, full_out_parent)
      end
    end
  end

  return index_label, stride_label, slice_mapping,
    parent_intersections, parent_barriers
end

local function upgrade_simultaneous_coherence(cx)
  local function make_simultaneous_coherence(edge)
    local from_label = cx.graph:node_label(edge.from_node)
    local to_label = cx.graph:node_label(edge.to_node)
    local from_type = from_label:is(flow.node.data) and from_label.region_type
    local to_type = to_label:is(flow.node.data) and to_label.region_type
    if is_any_privilege(edge) then
      if (from_type and std.is_list_of_regions(from_type)) or
        (to_type and std.is_list_of_regions(to_type))
      then
        -- Lists are simlutaneous.
        local coherence = flow.coherence_kind.Simultaneous {}
        local flag = flow.flag_kind.NoFlag {}
        if (from_type and from_type:list_depth() > 1) or
          (to_type and to_type:list_depth() > 1)
        then
          -- Intersections are no access.
          flag = flow.flag_kind.NoAccessFlag {}
        end
        return edge.label {
          coherence = coherence,
          flag = flag,
        }
      end
    end
  end

  cx.graph:map_edges(make_simultaneous_coherence)
end

local function downgrade_simultaneous_coherence(cx)
  local function make_exclusive_coherence(edge)
    if is_any_privilege(edge) then
      return edge.label {
        coherence = flow.coherence_kind.Exclusive {},
        flag = flow.flag_kind.NoFlag {},
      }
    end
  end

  cx.graph:map_edges(make_exclusive_coherence)
end

local function make_block(cx, block, mapping, span)
  local label = flow.node.ctrl.Block {
    block = block,
    annotations = ast.default_annotations(),
    span = span,
  }
  local nid = cx.graph:add_node(label)
  flow_summarize_subgraph.entry(cx.graph, nid, mapping)
  return nid
end

local function make_distribution_loop(cx, block, shard_index, shard_stride,
                                      global_vars, slice_mapping, span)
  local label = flow.node.ctrl.ForNum {
    symbol = shard_index.value.value,
    block = block,
    annotations = ast.default_annotations(),
    span = span,
  }
  local nid = cx.graph:add_node(label)
  flow_summarize_subgraph.entry(cx.graph, nid, slice_mapping)

  -- Add loop bounds.
  local zero_nid = cx.graph:add_node(flow.node.Constant {
    value = make_constant(0, global_vars.nparts.value.expr_type, span)
  })
  -- The node for global `nparts` should already been added as a predecessor
  -- to the ForNum node because it's used in the loop body.
  local upper_bound_nid =
    find_matching_input(cx, nid, global_vars.nparts.region_type) or
    cx.graph:add_node(global_vars.nparts)
  cx.graph:add_edge(flow.edge.Read(flow.default_mode()),
    zero_nid, cx.graph:node_result_port(zero_nid),
    nid, 1)
  cx.graph:add_edge(flow.edge.Read(flow.default_mode()),
    upper_bound_nid, cx.graph:node_result_port(upper_bound_nid),
    nid, 2)

  -- Add loop stride.
  if shard_stride:is(flow.node.Constant) then
    local stride_nid = cx.graph:add_node(shard_stride)
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()), stride_nid, cx.graph:node_result_port(stride_nid),
      nid, 3)
  else
    assert(false)
  end

  return nid
end

local function make_must_epoch(cx, block, annotations, span)
  -- Wrap block in a must epoch and return it.

  -- This gets a bit tricky, because the must epoch will use
  -- simultaneous coherence, and there might in general be other tasks
  -- at the same scope also using simultaneous coherence which would
  -- result in undesired reordering or concurrency. To avoid this,
  -- issue a fence immediately before and immediately after the must
  -- epoch. This essentially forces the must epoch to execute alone.

  -- It would be cleaner in some ways to wrap this in an inner task
  -- with exclusive coherence on all region parameters, though I'm
  -- worried that would have a substantially higher risk of
  -- introducing performance bugs because it would force the runtime
  -- to build virtual instances for all regions.

  local outer_cx = cx:new_graph_scope(flow.empty_graph(cx.tree))
  local inner_label = flow.node.ctrl.MustEpoch {
    block = block,
    annotations = annotations { spmd = ast.annotation.Forbid { value = false } },
    span = span,
  }
  local inner_nid = outer_cx.graph:add_node(inner_label)
  flow_summarize_subgraph.entry(outer_cx.graph, inner_nid, {})

  local fence_label = flow.node.Opaque {
    action = ast.typed.stat.Expr {
      expr = ast.typed.expr.Call {
        fn = ast.typed.expr.Function {
          value = std.c.legion_runtime_issue_execution_fence,
          expr_type = std.c.legion_runtime_issue_execution_fence:gettype(),
          annotations = ast.default_annotations(),
          span = span,
        },
        args = terralib.newlist({
          ast.typed.expr.RawRuntime {
            expr_type = std.c.legion_runtime_t,
            annotations = ast.default_annotations(),
            span = span,
          },
          ast.typed.expr.RawContext {
            expr_type = std.c.legion_context_t,
            annotations = ast.default_annotations(),
            span = span,
          },
        }),
        conditions = terralib.newlist({}),
        expr_type = terralib.types.unit,
        annotations = ast.default_annotations(),
        span = span,
      },
      annotations = ast.default_annotations(),
      span = span,
    }
  }
  local before_nid = outer_cx.graph:add_node(fence_label)
  local after_nid = outer_cx.graph:add_node(fence_label)

  outer_cx.graph:add_edge(
    flow.edge.HappensBefore {},
    before_nid, outer_cx.graph:node_sync_port(before_nid),
    inner_nid, outer_cx.graph:node_sync_port(inner_nid))
  outer_cx.graph:add_edge(
    flow.edge.HappensBefore {},
    inner_nid, outer_cx.graph:node_sync_port(inner_nid),
    after_nid, outer_cx.graph:node_sync_port(after_nid))

  local outer_label = flow.node.ctrl.Block {
    block = outer_cx.graph,
    annotations = ast.default_annotations(),
    span = span,
  }

  local outer_nid = cx.graph:add_node(outer_label)
  flow_summarize_subgraph.entry(cx.graph, outer_nid, {})
  return outer_nid
end

local function find_match_incoming(cx, predicate, nid)
  local edges = cx.graph:incoming_edges(nid)
  for _, edge in ipairs(edges) do
    if isnt_happens_before(edge) then
      local other_nid = edge.from_node
      local label = cx.graph:node_label(other_nid)
      if predicate(other_nid, label) then
        return other_nid
      end
    end
  end
end

local function find_match_outgoing(cx, predicate, nid)
  local edges = cx.graph:outgoing_edges(nid)
  for _, edge in ipairs(edges) do
    if isnt_happens_before(edge) then
      local other_nid = edge.to_node
      local label = cx.graph:node_label(other_nid)
      if predicate(other_nid, label) then
        return other_nid
      end
    end
  end
end

local function find_nid_mapping(cx, old_loop, new_loop,
                                intersection_types,
                                barriers_empty_in, barriers_empty_out,
                                barriers_full_in, barriers_full_out,
                                collective_types, start_barrier,
                                global_vars, mapping)
  local function subtree_contains(old_type, new_type)
    if (mapping[old_type] or old_type) == new_type then
      return true
    end
    for _, child_type in ipairs(cx.tree:children(old_type)) do
      if subtree_contains(child_type, new_type) then
        return true
      end
    end
  end

  local function matches(new_label)
    return function(nid, label)
      if label:is(flow.node.data) then
        if label.field_path == new_label.field_path then
          return subtree_contains(label.region_type, new_label.region_type)
        end
      end
    end
  end

  local new_inputs = cx.graph:incoming_edges(new_loop)
  local input_nid_mapping = data.new_recursive_map(2)
  for _, edge in ipairs(new_inputs) do
    local new_input_nid = edge.from_node
    local new_input = cx.graph:node_label(new_input_nid)
    if new_input:is(flow.node.data) then
      local old_input_nid = find_match_incoming(
        cx, matches(new_input), old_loop)
      if not old_input_nid then
        assert(intersection_types[new_input.region_type] or
                 barriers_empty_in[new_input.region_type] or
                 barriers_empty_out[new_input.region_type] or
                 barriers_full_in[new_input.region_type] or
                 barriers_full_out[new_input.region_type] or
                 collective_types[new_input.region_type] or
                 new_input.region_type == start_barrier.region_type or
                 new_input.region_type == global_vars.nparts.region_type or
                 new_input.region_type == global_vars.part_indices.region_type)
          -- Skip intersections, phase barriers, collectives, and newly created
          -- global variables.
      else
        input_nid_mapping[new_input.region_type][new_input.field_path][
          old_input_nid] = new_input_nid
      end
    end
  end
  local output_nid_mapping = data.new_recursive_map(2)
  local new_outputs = cx.graph:outgoing_edges(new_loop)
  for _, edge in ipairs(new_outputs) do
    local new_output_nid = edge.to_node
    local new_output = cx.graph:node_label(new_output_nid)
    if new_output:is(flow.node.data) then
      local old_output_nid = find_match_outgoing(
        cx, matches(new_output), old_loop)
      if not old_output_nid then
        assert(intersection_types[new_output.region_type] or
                 barriers_empty_in[new_output.region_type] or
                 barriers_empty_out[new_output.region_type] or
                 barriers_full_in[new_output.region_type] or
                 barriers_full_out[new_output.region_type] or
                 collective_types[new_input.region_type] or
                 new_input.region_type == start_barrier.region_type)
        -- Skip intersections, phase barriers, collectives.
      else
        output_nid_mapping[new_output.region_type][new_output.field_path][
          old_output_nid] = new_output_nid
      end
    end
  end
  return input_nid_mapping, output_nid_mapping
end

local function find_partition_nids(cx, region_type, need_copy, partitions,
                                   force_open)
  local old_nids = data.newmap()
  local new_nids = data.newmap()
  for field_path, nid_mapping in need_copy:items() do
    for old_nid, new_nid in nid_mapping:items() do
      assert(not old_nids[field_path] and not new_nids[field_path])
      old_nids[field_path] = old_nid
      new_nids[field_path] = new_nid
    end
  end

  local partition_nids = data.newmap()
  for field_path, old_nid in old_nids:items() do
    local old_label = cx.graph:node_label(old_nid)
    if not old_label:is(flow.node.data.Partition) then
      local partition_label = partitions[region_type][field_path]
      if force_open then
        local open_nid = maybe(data.filter(
          function(nid) return cx.graph:node_label(nid):is(flow.node.Open) end,
          cx.graph:outgoing_read_set(old_nid)))
        if not open_nid then
          open_nid = cx.graph:add_node(flow.node.Open {})
          cx.graph:add_edge(
            flow.edge.Read(flow.default_mode()),
            old_nid, cx.graph:node_result_port(old_nid),
            open_nid, cx.graph:node_available_port(open_nid))
        end

        local opened_nid = find_matching_output(
          cx, open_nid, partition_label.region_type, partition_label.field_path)
        if not opened_nid then
          opened_nid = cx.graph:add_node(partition_label)
          cx.graph:add_edge(
            flow.edge.Write(flow.default_mode()),
            open_nid, cx.graph:node_result_port(open_nid),
            opened_nid, 0)
        end
        partition_nids[field_path] = opened_nid
      else
        partition_nids[field_path] = cx.graph:add_node(partition_label)
      end
    else
      partition_nids[field_path] = old_nid
    end
  end

  -- Grab the first of each for convenience.
  local first_partition_label, first_new_label
  for field_path, partition_nid in partition_nids:items() do
    first_partition_label = cx.graph:node_label(partition_nid)
    break
  end
  for field_path, new_nid in new_nids:items() do
    first_new_label = cx.graph:node_label(new_nid)
    break
  end
  assert(first_partition_label and first_new_label)
  assert(first_partition_label:is(flow.node.data.Partition))

  return old_nids, new_nids, partition_nids, first_partition_label, first_new_label
end

local function find_cross_product_nids(cx, region_type, need_copy, partitions,
                                       always_create)
  local old_nids = data.newmap()
  local new_nids = data.newmap()
  for field_path, nid_mapping in need_copy:items() do
    for old_nid, new_nid in nid_mapping:items() do
      assert(not old_nids[field_path] and not new_nids[field_path])
      old_nids[field_path] = old_nid
      new_nids[field_path] = new_nid
    end
  end

  local partition_nids = data.newmap()
  for field_path, old_nid in old_nids:items() do
    local old_label = cx.graph:node_label(old_nid)
    if not old_label:is(flow.node.data.CrossProduct) then
      assert(false)
      -- local partition_label = partitions[region_type][field_path]
      -- partition_nids[field_path] = cx.graph:add_node(partition_label)
    elseif always_create then
      assert(false)
      -- partition_nids[field_path] = cx.graph:add_node(old_label)
    else
      partition_nids[field_path] = old_nid
    end
  end

  -- Grab the first of each for convenience.
  local first_partition_label, first_new_label
  for field_path, partition_nid in partition_nids:items() do
    first_partition_label = cx.graph:node_label(partition_nid)
    break
  end
  for field_path, new_nid in new_nids:items() do
    first_new_label = cx.graph:node_label(new_nid)
    break
  end
  assert(first_partition_label and first_new_label)
  assert(first_partition_label:is(flow.node.data.CrossProduct))

  return old_nids, new_nids, partition_nids, first_partition_label, first_new_label
end

local function find_root_region(cx, nid)
  local label = cx.graph:node_label(nid)
  while not label:is(flow.node.data.Region) do
    local writers = cx.graph:incoming_write_set(nid)
    assert(#writers == 1)
    local writer = writers[1]

    local reads = cx.graph:filter_immediate_predecessors_by_edges(
      function(edge, from_label)
        local ancestor = cx.tree:lowest_common_ancestor(from_label.region_type, label.region_type)
        return is_exact_match(from_label, ancestor, from_label.field_path) and
          is_read(edge)
      end,
      writer)
    assert(#reads == 1)
    local read = reads[1]

    nid = read
    label = cx.graph:node_label(nid)
  end
  return nid
end

local function find_close_region(cx, nid, loop)
  local closes = cx.graph:filter_immediate_successors_by_edges(
    function(edge, to_label)
      return to_label:is(flow.node.Close) and is_read(edge)
    end,
    nid)
  if #closes < 1 then
    return
  end
  assert(#closes == 1)
  local close = closes[1]

  if cx.graph:reachable(loop, close) then
    return
  end

  local writes = cx.graph:outgoing_write_set(close)
  assert(#writes == 1)
  return writes[1]
end

local function issue_zipped_copy_interior(
    cx, src_nids, dst_in_nids, dst_out_nids,
    src_type, dst_type, nparts_nid, part_indices_nid, span)

  local part_indices_label = cx.graph:node_label(part_indices_nid)

  -- Build the copy loop.
  local block_cx = cx:new_graph_scope(flow.empty_graph(cx.tree))

  local index_type = std.rawref(&int)
  local index_symbol = std.newsymbol(int, "index")
  local index_label = make_variable_label(
    block_cx, index_symbol, int, span)
  local index_nid = block_cx.graph:add_node(index_label)

  -- Exterior of the loop:
  local copy_loop = flow.node.ctrl.ForNum {
    symbol = index_symbol,
    block = block_cx.graph,
    annotations = ast.default_annotations(),
    span = span,
  }
  local copy_loop_nid = cx.graph:add_node(copy_loop)

  local zero_nid = cx.graph:add_node(flow.node.Constant {
    value = make_constant(0, int, span),
  })
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    zero_nid, cx.graph:node_result_port(zero_nid),
    copy_loop_nid, 1)
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    nparts_nid, cx.graph:node_result_port(nparts_nid),
    copy_loop_nid, 2)
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    part_indices_nid, cx.graph:node_result_port(part_indices_nid),
    copy_loop_nid, cx.graph:node_available_port(copy_loop_nid))

  local copy_loop_src_port = cx.graph:node_available_port(copy_loop_nid)
  for field_path, src_nid in src_nids:items() do
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()), src_nid, cx.graph:node_result_port(src_nid),
      copy_loop_nid, copy_loop_src_port)
  end
  local copy_loop_dst_port = cx.graph:node_available_port(copy_loop_nid)
  for field_path, dst_in_nid in dst_in_nids:items() do
    local dst_out_nid = dst_out_nids[field_path]
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()), dst_in_nid, cx.graph:node_result_port(dst_in_nid),
      copy_loop_nid, copy_loop_dst_port)
    cx.graph:add_edge(
      flow.edge.Write(flow.default_mode()), copy_loop_nid, copy_loop_dst_port,
      dst_out_nid, cx.graph:node_available_port(dst_out_nid))
  end

  -- Interior of the loop:

  -- The regions we copy to/from come from either a list or a partition.  This
  -- function returns the node that should be used to index into the
  -- list/partition: `index` for list, `part_indices[index]` for partition.
  local function get_index_access_index_nid(cx, value_type, part_indices_label,
                                            index_nid)
    assert(cx.graph:has_node(index_nid))
    if std.is_list(value_type) then
      -- Just use `index_nid` to index into the List.
      return index_nid
    elseif std.is_partition(value_type) then
      -- Otherwise, we need `part_indices[index]` to index into a Partition.
      local part_index_type =
        std.as_read(part_indices_label.value.expr_type).element_type
      local compute_part_index_nid = cx.graph:add_node(
        flow.node.IndexAccess {
          expr_type = part_index_type,
          annotations = ast.default_annotations(),
          span = span,
        })
      -- Make node for `part_indices` in loop body block.
      local part_indices_nid_block = cx.graph:add_node(part_indices_label)
      block_cx.graph:add_edge(
        flow.edge.Read(flow.default_mode()),
        part_indices_nid_block, cx.graph:node_result_port(part_indices_nid_block),
        compute_part_index_nid, 1)
      block_cx.graph:add_edge(
        flow.edge.Read(flow.default_mode()),
        index_nid, cx.graph:node_result_port(index_nid),
        compute_part_index_nid, 2)
      local part_index_nid = block_cx.graph:add_node(
        make_variable_label(block_cx, std.newsymbol(part_index_type),
          part_index_type, span) { fresh = true })
      block_cx.graph:add_edge(
        flow.edge.Write(flow.default_mode()),
        compute_part_index_nid, block_cx.graph:node_result_port(compute_part_index_nid),
        part_index_nid, 0)
      return part_index_nid
    else
      assert(false)
    end
  end

  local block_src_i_type = src_type:subregion_dynamic()
  if std.is_partition(src_type) then
    std.add_constraint(
      cx.tree, src_type, src_type:parent_region(), std.subregion, false)
    std.add_constraint(
      cx.tree, block_src_i_type, src_type, std.subregion, false)
    cx.tree.region_universe[block_src_i_type] = true
  end

  local block_src_nids = data.newmap()
  local block_src_i_nids = data.newmap()
  for field_path, src_nid in src_nids:items() do
    local src_label = cx.graph:node_label(src_nid)
    block_src_nids[field_path] = block_cx.graph:add_node(src_label)
    block_src_i_nids[field_path] = block_cx.graph:add_node(
      src_label {
        value = src_label.value {
          value = std.newsymbol(block_src_i_type),
          expr_type = block_src_i_type,
        },
      })
  end

  local block_dst_i_type = dst_type:subregion_dynamic()
  if std.is_partition(dst_type) then
    std.add_constraint(
      cx.tree, dst_type, dst_type:parent_region(), std.subregion, false)
    std.add_constraint(
      cx.tree, block_dst_i_type, dst_type, std.subregion, false)
    cx.tree.region_universe[block_dst_i_type] = true
  end

  local block_dst_in_nids = data.newmap()
  local block_dst_in_i_nids = data.newmap()
  local block_dst_out_i_nids = data.newmap()
  for field_path, dst_in_nid in dst_in_nids:items() do
    local dst_in_label = cx.graph:node_label(dst_in_nid)
    block_dst_in_nids[field_path] = block_cx.graph:add_node(dst_in_label)

    local block_dst_i = flow.node.data.Region(dst_in_label) {
      value = dst_in_label.value {
        value = std.newsymbol(block_dst_i_type),
        expr_type = block_dst_i_type,
      },
    }
    block_dst_in_i_nids[field_path] = block_cx.graph:add_node(block_dst_i)
    block_dst_out_i_nids[field_path] = block_cx.graph:add_node(block_dst_i)
  end

  local block_index_src_nid = block_cx.graph:add_node(
    flow.node.IndexAccess {
      expr_type = block_src_i_type,
      annotations = ast.default_annotations(),
      span = span,
    })
  for field_path, block_src_nid in block_src_nids:items() do
    block_cx.graph:add_edge(
      flow.edge.None(flow.default_mode()),
      block_src_nid, block_cx.graph:node_result_port(block_src_nid),
      block_index_src_nid, 1)
  end
  local src_index_nid = get_index_access_index_nid(block_cx, src_type,
    part_indices_label, index_nid)
  block_cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    src_index_nid, block_cx.graph:node_result_port(src_index_nid),
    block_index_src_nid, 2)
  for field_path, block_src_i_nid in block_src_i_nids:items() do
    block_cx.graph:add_edge(
      flow.edge.Name {},
      block_index_src_nid,
      block_cx.graph:node_result_port(block_index_src_nid),
      block_src_i_nid, block_cx.graph:node_available_port(block_src_i_nid))
  end

  local block_index_dst_in_nid = block_cx.graph:add_node(
    flow.node.IndexAccess {
      expr_type = block_dst_i_type,
      annotations = ast.default_annotations(),
      span = span,
    })
  for field_path, block_dst_in_nid in block_dst_in_nids:items() do
    block_cx.graph:add_edge(
      flow.edge.None(flow.default_mode()),
      block_dst_in_nid, block_cx.graph:node_result_port(block_dst_in_nid),
      block_index_dst_in_nid, 1)
  end
  local dst_index_nid = get_index_access_index_nid(block_cx, dst_type,
    part_indices_label, index_nid)
  block_cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    dst_index_nid, block_cx.graph:node_result_port(dst_index_nid),
    block_index_dst_in_nid, 2)
  for field_path, block_dst_in_i_nid in
    block_dst_in_i_nids:items()
  do
    block_cx.graph:add_edge(
      flow.edge.Name {},
      block_index_dst_in_nid,
      block_cx.graph:node_result_port(block_index_dst_in_nid),
      block_dst_in_i_nid,
      block_cx.graph:node_available_port(block_dst_in_i_nid))
  end

  -- Copy data to the opened partition.
  local field_paths = src_nids:map_list(function(k) return k end)
  local block_copy = flow.node.Copy {
    src_field_paths = field_paths,
    dst_field_paths = field_paths,
    op = false,
    annotations = ast.default_annotations(),
    span = span,
  }
  local block_copy_nid = block_cx.graph:add_node(block_copy)
  for field_path, block_src_i_nid in block_src_i_nids:items() do
    block_cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()),
      block_src_i_nid, block_cx.graph:node_result_port(block_src_i_nid),
      block_copy_nid, 1)
  end
  for field_path, block_dst_in_i_nid in
    block_dst_in_i_nids:items()
  do
    local block_dst_out_i_nid = block_dst_out_i_nids[field_path]
    block_cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()),
      block_dst_in_i_nid,
      block_cx.graph:node_result_port(block_dst_in_i_nid),
      block_copy_nid, 2)
    block_cx.graph:add_edge(
      flow.edge.Write(flow.default_mode()), block_copy_nid, 2,
      block_dst_out_i_nid,
      block_cx.graph:node_result_port(block_dst_out_i_nid))
  end

  return copy_loop_nid
end


local function issue_zipped_copy(cx, src_nids, dst_in_nids, dst_out_nids,
                                 src_type, dst_type, nparts_nid,
                                 part_indices_nid, shadow, span)
  local shadow_label
  if shadow then
    for field_path, label in shadow:items() do
      shadow_label = label
    end
    assert(shadow_label and std.is_partition(shadow_label.region_type))
  end

  local part_type = std.is_partition(src_type) and src_type or dst_type
  if not shadow or
    not permit_dominance_via_complete(cx, part_type, shadow_label.region_type)
  then
    return issue_zipped_copy_interior(
      cx, src_nids, dst_in_nids, dst_out_nids,
      src_type, dst_type, nparts_nid, part_indices_nid, span)
  else

    -- Build the outer conditional.
    local block_cx = cx:new_graph_scope(flow.empty_graph(cx.tree))

    -- Check the conditional.
    local cond_type = std.rawref(&int)
    local cond_symbol = std.newsymbol(int, "complete_cond")
    local cond_label = make_variable_label(
      block_cx, cond_symbol, int, span)
    local cond_nid = cx.graph:add_node(cond_label)

    local check = flow.node.Opaque {
      action = ast.typed.stat.Var {
        symbols = terralib.newlist({cond_symbol}),
        types = terralib.newlist({int}),
        values = terralib.newlist({
            ast.typed.expr.Cast {
              fn = ast.typed.expr.Function {
                value = int,
                expr_type = {std.untyped} -> int,
                annotations = ast.default_annotations(),
                span = span,
              },
              arg = ast.typed.expr.Call {
                fn = ast.typed.expr.Function {
                  value = std.c.legion_index_partition_is_complete,
                  expr_type = std.c.legion_index_partition_is_complete:gettype(),
                  annotations = ast.default_annotations(),
                  span = span,
                },
                args = terralib.newlist({
                    ast.typed.expr.RawRuntime {
                      expr_type = std.c.legion_runtime_t,
                      annotations = ast.default_annotations(),
                      span = span,
                    },
                    ast.typed.expr.FieldAccess {
                      value = ast.typed.expr.RawValue {
                        value = shadow_label.value,
                        expr_type = std.c.legion_logical_partition_t,
                        annotations = ast.default_annotations(),
                        span = span,
                      },
                      field_name = "index_partition",
                      expr_type = std.c.legion_index_partition_t,
                      annotations = ast.default_annotations(),
                      span = span,
                    },
                }),
                conditions = terralib.newlist({}),
                expr_type = bool,
                annotations = ast.default_annotations(),
                span = span,
              },
              expr_type = int,
              annotations = ast.default_annotations(),
              span = span,
            },
        }),
        annotations = ast.default_annotations(),
        span = span,
      }
    }
    local check_nid = cx.graph:add_node(check)
    for field_path, src_nid in src_nids:items() do
      cx.graph:add_edge(
        flow.edge.HappensBefore {}, src_nid, cx.graph:node_sync_port(src_nid),
        check_nid, cx.graph:node_sync_port(check_nid))
    end
    cx.graph:add_edge(
      flow.edge.HappensBefore {}, check_nid, cx.graph:node_sync_port(check_nid),
      cond_nid, cx.graph:node_sync_port(cond_nid))

    local index_type = std.rawref(&int)
    local index_symbol = std.newsymbol(int, "complete_index")
    local index_label = make_variable_label(
      block_cx, index_symbol, int, span)
    local index_nid = block_cx.graph:add_node(index_label)

    -- Exterior of the loop:
    local copy_loop = flow.node.ctrl.ForNum {
      symbol = index_symbol,
      block = block_cx.graph,
      annotations = ast.default_annotations(),
      span = span,
    }
    local copy_loop_nid = cx.graph:add_node(copy_loop)
    local bound1_nid = cond_nid
    local bound2_nid = cx.graph:add_node(
      flow.node.Constant { value = make_constant(1, int, span) })
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()),
      bound1_nid, cx.graph:node_result_port(bound1_nid),
      copy_loop_nid, 1)
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()),
      bound2_nid, cx.graph:node_result_port(bound2_nid),
      copy_loop_nid, 2)
    local copy_loop_src_port = cx.graph:node_available_port(copy_loop_nid)
    for field_path, src_nid in src_nids:items() do
      cx.graph:add_edge(
        flow.edge.Read(flow.default_mode()), src_nid, cx.graph:node_result_port(src_nid),
        copy_loop_nid, copy_loop_src_port)
    end
    local copy_loop_dst_port = cx.graph:node_available_port(copy_loop_nid)
    for field_path, dst_in_nid in dst_in_nids:items() do
      local dst_out_nid = dst_out_nids[field_path]
      cx.graph:add_edge(
        flow.edge.Read(flow.default_mode()), dst_in_nid, cx.graph:node_result_port(dst_in_nid),
        copy_loop_nid, copy_loop_dst_port)
      cx.graph:add_edge(
        flow.edge.Write(flow.default_mode()), copy_loop_nid, copy_loop_dst_port,
        dst_out_nid, cx.graph:node_available_port(dst_out_nid))
    end

    -- Interior of loop:

    local inner_src_nids = data.newmap()
    for field_path, src_nid in src_nids:items() do
      local inner_src_nid = block_cx.graph:add_node(cx.graph:node_label(src_nid))
      inner_src_nids[field_path] = inner_src_nid
    end

    local inner_dst_in_nids = data.newmap()
    for field_path, dst_in_nid in dst_in_nids:items() do
      local inner_dst_in_nid = block_cx.graph:add_node(cx.graph:node_label(dst_in_nid))
      inner_dst_in_nids[field_path] = inner_dst_in_nid
    end

    local inner_dst_out_nids = data.newmap()
    for field_path, dst_out_nid in dst_out_nids:items() do
      local inner_dst_out_nid = block_cx.graph:add_node(cx.graph:node_label(dst_out_nid))
      inner_dst_out_nids[field_path] = inner_dst_out_nid
    end

    -- Propagate `nparts` and `part_indices` to inner graph.
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()),
      nparts_nid, cx.graph:node_result_port(nparts_nid),
      copy_loop_nid, cx.graph:node_available_port(copy_loop_nid))
    cx.graph:add_edge(
      flow.edge.Read(flow.default_mode()),
      part_indices_nid, cx.graph:node_result_port(part_indices_nid),
      copy_loop_nid, cx.graph:node_available_port(copy_loop_nid))
    local inner_nparts_nid = block_cx.graph:add_node(cx.graph:node_label(nparts_nid))
    local inner_part_indices_nid = block_cx.graph:add_node(cx.graph:node_label(part_indices_nid))

    return issue_zipped_copy_interior(
      block_cx, inner_src_nids, inner_dst_in_nids, inner_dst_out_nids,
      src_type, dst_type, inner_nparts_nid, inner_part_indices_nid, span)
  end
end

local function issue_input_copies_partition(
    cx, region_type, need_copy, elide, shadow, partitions, old_loop,
    nparts_nid, part_indices_nid, opened_nids)
  assert(std.is_list_of_regions(region_type))
  local old_nids, new_nids, partition_nids, first_partition_label, first_new_label =
    find_partition_nids(
      cx, region_type, need_copy, partitions, true)

  -- Record the opened partitions.
  opened_nids[region_type] = partition_nids

  -- Name the intermediate list (before it has valid data).
  local name_nids
  if elide then
    name_nids = new_nids
  else
    name_nids = data.newmap()
    for field_path, new_nid in new_nids:items() do
      local new_label = cx.graph:node_label(new_nid)
      local name_nid = cx.graph:add_node(new_label)
      name_nids[field_path] = name_nid
    end
  end

  -- Duplicate the partition.
  local slice = ast.typed.expr.ListDuplicatePartition
  if elide then
    slice = ast.typed.expr.ListSlicePartition
  end

  local indices = cx.graph:node_label(part_indices_nid).value
  local duplicated = slice {
    partition = first_partition_label.value,
    indices = indices,
    expr_type = std.as_read(first_new_label.value.expr_type),
    annotations = ast.default_annotations(),
    span = first_partition_label.value.span,
  }
  local duplicate = flow.node.Opaque {
    action = ast.typed.stat.Var {
      symbols = terralib.newlist({ first_new_label.value.value }),
      types = terralib.newlist({ region_type }),
      values = terralib.newlist({ duplicated }),
      annotations = ast.default_annotations(),
      span = first_partition_label.value.span,
    }
  }
  local duplicate_nid = cx.graph:add_node(duplicate)
  local duplicate_port = cx.graph:node_available_port(duplicate_nid)
  for field_path, partition_nid in partition_nids:items() do
    cx.graph:add_edge(
      flow.edge.None(flow.default_mode()), partition_nid, cx.graph:node_result_port(),
      duplicate_nid, duplicate_port)
  end
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    part_indices_nid, cx.graph:node_result_port(part_indices_nid),
    duplicate_nid, cx.graph:node_available_port(duplicate_nid))
  for field_path, name_nid in name_nids:items() do
    cx.graph:add_edge(
      flow.edge.HappensBefore {},
      duplicate_nid, cx.graph:node_sync_port(duplicate_nid),
      name_nid, cx.graph:node_sync_port(name_nid))
  end

  if not elide then
    local copy_nid = issue_zipped_copy(
      cx, partition_nids, name_nids, new_nids,
      std.as_read(first_partition_label.value.expr_type), region_type,
      nparts_nid, part_indices_nid, shadow, first_new_label.value.span)
    cx.graph:copy_incoming_edges(is_happens_before, old_loop, copy_nid)
  end
end

local function issue_input_copies_cross_product(
    cx, region_type, need_copy, elide, shadow, partitions, old_loop,
    nparts_nid, part_indices_nid, opened_nids)
  assert(std.is_list_of_partitions(region_type))
  local old_nids, new_nids, product_nids, first_product_label, first_new_label =
    find_cross_product_nids(
      cx, region_type, need_copy, partitions, false)

  -- Right now we're not in a position to issue copies. So the node
  -- we're producing had better not get read.
  for field_path, old_nid in old_nids:items() do
    local product_nid = product_nids[field_path]
    local new_nid = new_nids[field_path]
    assert(old_nid == product_nid)
    assert(#cx.graph:outgoing_read_set(new_nid) == 0)
  end

  -- Duplicate the product.
  local part_indices_label = cx.graph:node_label(part_indices_nid)
  local duplicate = flow.node.Opaque {
    action = ast.typed.stat.Var {
      symbols = terralib.newlist({
          first_new_label.value.value,
      }),
      types = terralib.newlist({
          region_type,
      }),
      values = terralib.newlist({
          ast.typed.expr.ListSliceCrossProduct {
            product = first_product_label.value,
            indices = part_indices_label.value,
            expr_type = std.as_read(first_new_label.value.expr_type),
            annotations = ast.default_annotations(),
            span = first_product_label.value.span,
          },
      }),
      annotations = ast.default_annotations(),
      span = first_product_label.value.span,
    }
  }
  local duplicate_nid = cx.graph:add_node(duplicate)
  local duplicate_port = cx.graph:node_available_port(duplicate_nid)
  for field_path, product_nid in product_nids:items() do
    cx.graph:add_edge(
      flow.edge.None(flow.default_mode()), product_nid, cx.graph:node_result_port(),
      duplicate_nid, duplicate_port)
  end
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    part_indices_nid, cx.graph:node_result_port(part_indices_nid),
    duplicate_nid, cx.graph:node_available_port(duplicate_nid))
  for field_path, new_nid in new_nids:items() do
    cx.graph:add_edge(
      flow.edge.HappensBefore {},
      duplicate_nid, cx.graph:node_sync_port(duplicate_nid),
      new_nid, cx.graph:node_sync_port(new_nid))
  end
end

local function issue_input_copies(
    cx, region_type, need_copy, elide, shadow, partitions, old_loop,
    nparts_nid, part_indices_nid, opened_nids)
  if std.is_list_of_regions(region_type) then
    issue_input_copies_partition(
      cx, region_type, need_copy, elide, shadow, partitions, old_loop,
      nparts_nid, part_indices_nid, opened_nids)
  elseif std.is_list_of_partitions(region_type) then
    issue_input_copies_cross_product(
      cx, region_type, need_copy, elide, shadow, partitions, old_loop,
      nparts_nid, part_indices_nid, opened_nids)
  else
    assert(false)
  end
end

local function issue_output_copies_partition(
    cx, region_type, need_copy, elide, shadow, partitions, old_loop,
    nparts_nid, part_indices_nid, opened_nids)
  assert(std.is_list_of_regions(region_type))
  local old_nids, new_nids, partition_nids, first_partition_label, first_new_label =
    find_partition_nids(
      cx, region_type, need_copy, partitions, false)

  local opened_partition_nids = data.newmap()
  for field_path, old_nid in old_nids:items() do
    opened_partition_nids[field_path] = opened_nids[region_type][field_path]
  end

  -- Close any open partitions back into the original region.
  for field_path, old_nid in old_nids:items() do
    local partition_nid = partition_nids[field_path]
    if partition_nid ~= old_nid then
      local close_nid = cx.graph:add_node(flow.node.Close {})
      cx.graph:add_edge(
        flow.edge.Read(flow.default_mode()), partition_nid, cx.graph:node_result_port(partition_nid),
        close_nid, cx.graph:node_available_port(close_nid))
      cx.graph:add_edge(
        flow.edge.Write(flow.default_mode()), close_nid, cx.graph:node_result_port(close_nid),
        old_nid, cx.graph:node_available_port(old_nid))
    end
  end

  if not elide then
    local copy_nid = issue_zipped_copy(
      cx, new_nids, opened_partition_nids, partition_nids,
      region_type, std.as_read(first_partition_label.value.expr_type),
      nparts_nid, part_indices_nid, shadow, first_new_label.value.span)
    cx.graph:copy_outgoing_edges(is_happens_before, old_loop, copy_nid)
  else
    for field_path, new_nid in new_nids:items() do
      local opened_partition_nid = opened_partition_nids[field_path]
      local partition_nid = partition_nids[field_path]

      local close_nid = cx.graph:add_node(flow.node.Close {})
      cx.graph:add_edge(
        flow.edge.Read(flow.default_mode()), opened_partition_nid, cx.graph:node_result_port(opened_partition_nid),
        close_nid, cx.graph:node_available_port(close_nid))
      cx.graph:add_edge(
        flow.edge.Write(flow.default_mode()), close_nid, cx.graph:node_result_port(close_nid),
        partition_nid, cx.graph:node_available_port(partition_nid))

      cx.graph:add_edge(
        flow.edge.HappensBefore {}, new_nid, cx.graph:node_sync_port(new_nid),
        close_nid, cx.graph:node_sync_port(close_nid))
    end
  end
end

local function issue_output_copies_cross_product(
    cx, region_type, need_copy, elide, shadow, partitions, old_loop,
    nparts_nid, part_indices_nid, opened_nids)
  assert(std.is_list_of_partitions(region_type))
  assert(false)
end

local function issue_output_copies(
    cx, region_type, need_copy, elide, shadow, partitions, old_loop,
    nparts_nid, part_indices_nid, opened_nids)
  if std.is_list_of_regions(region_type) then
    issue_output_copies_partition(
      cx, region_type, need_copy, elide, shadow, partitions, old_loop,
      nparts_nid, part_indices_nid, opened_nids)
  elseif std.is_list_of_partitions(region_type) then
    issue_output_copies_cross_product(
      cx, region_type, need_copy, elide, shadow, partitions, old_loop,
      nparts_nid, part_indices_nid, opened_nids)
  else
    assert(false)
  end
end

local function issue_intersection_creation(cx, intersection_nids,
                                           lhs_nid, rhs_nid)
  local first_intersection = cx.graph:node_label(intersection_nids[1])
  local lhs = cx.graph:node_label(lhs_nid)
  local rhs = cx.graph:node_label(rhs_nid)

  local cross_product = flow.node.Opaque {
    action = ast.typed.stat.Var {
      symbols = terralib.newlist({
          first_intersection.value.value,
      }),
      types = terralib.newlist({
          first_intersection.region_type,
      }),
      values = terralib.newlist({
          ast.typed.expr.ListCrossProduct {
            lhs = lhs.value,
            rhs = rhs.value,
            shallow = true,
            expr_type = std.as_read(first_intersection.value.expr_type),
            annotations = ast.default_annotations(),
            span = first_intersection.value.span,
          },
      }),
      annotations = ast.default_annotations(),
      span = first_intersection.value.span,
    }
  }
  local cross_product_nid = cx.graph:add_node(cross_product)
  cx.graph:add_edge(
    flow.edge.None(flow.default_mode()), lhs_nid, cx.graph:node_result_port(lhs_nid),
    cross_product_nid, cx.graph:node_available_port(cross_product_nid))
  cx.graph:add_edge(
    flow.edge.None(flow.default_mode()), rhs_nid, cx.graph:node_result_port(rhs_nid),
    cross_product_nid, cx.graph:node_available_port(cross_product_nid))
  for _, intersection_nid in ipairs(intersection_nids) do
    cx.graph:add_edge(
      flow.edge.HappensBefore {},
      cross_product_nid, cx.graph:node_sync_port(cross_product_nid),
      intersection_nid, cx.graph:node_sync_port(intersection_nid))
  end
end

local function issue_barrier_creation(cx, rhs_nid, intersection_nid,
                                      barrier_in_nid, barrier_out_nid)
  local rhs = cx.graph:node_label(rhs_nid)
  local intersection = cx.graph:node_label(intersection_nid)
  local barrier_in = cx.graph:node_label(barrier_in_nid)
  local barrier_out = cx.graph:node_label(barrier_out_nid)

  local list_barriers = flow.node.Opaque {
    action = ast.typed.stat.Var {
      symbols = terralib.newlist({
          barrier_out.value.value,
      }),
      types = terralib.newlist({
          std.as_read(barrier_out.value.expr_type),
      }),
      values = terralib.newlist({
          ast.typed.expr.ListPhaseBarriers {
            product = intersection.value,
            expr_type = std.as_read(barrier_out.value.expr_type),
            annotations = ast.default_annotations(),
            span = barrier_out.value.span,
          },
      }),
      annotations = ast.default_annotations(),
      span = barrier_out.value.span,
    }
  }
  local list_barriers_nid = cx.graph:add_node(list_barriers)
  cx.graph:add_edge(
    flow.edge.None(flow.default_mode()), intersection_nid, cx.graph:node_result_port(intersection_nid),
    list_barriers_nid, cx.graph:node_available_port(list_barriers_nid))
  cx.graph:add_edge(
    flow.edge.HappensBefore {},
    list_barriers_nid, cx.graph:node_sync_port(list_barriers_nid),
    barrier_out_nid, cx.graph:node_sync_port(barrier_out_nid))

  local list_invert = flow.node.Opaque {
    action = ast.typed.stat.Var {
      symbols = terralib.newlist({
          barrier_in.value.value,
      }),
      types = terralib.newlist({
          std.as_read(barrier_in.value.expr_type),
      }),
      values = terralib.newlist({
          ast.typed.expr.ListInvert {
            rhs = rhs.value,
            product = intersection.value,
            barriers = barrier_out.value,
            expr_type = std.as_read(barrier_in.value.expr_type),
            annotations = ast.default_annotations(),
            span = barrier_in.value.span,
          },
      }),
      annotations = ast.default_annotations(),
      span = barrier_in.value.span,
    }
  }
  local list_invert_nid = cx.graph:add_node(list_invert)
  cx.graph:add_edge(
    flow.edge.None(flow.default_mode()), rhs_nid, cx.graph:node_result_port(rhs_nid),
    list_invert_nid, cx.graph:node_available_port(list_invert_nid))
  cx.graph:add_edge(
    flow.edge.None(flow.default_mode()), intersection_nid, cx.graph:node_result_port(intersection_nid),
    list_invert_nid, cx.graph:node_available_port(list_invert_nid))
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()), barrier_out_nid, cx.graph:node_result_port(barrier_out_nid),
    list_invert_nid, cx.graph:node_available_port(list_invert_nid))
  cx.graph:add_edge(
    flow.edge.HappensBefore {},
    list_invert_nid, cx.graph:node_sync_port(list_invert_nid),
    barrier_in_nid, cx.graph:node_sync_port(barrier_in_nid))
end

local function issue_collective_creation(cx, loop_nid, collective_nid, op, nparts_nid)
  local collective = cx.graph:node_label(collective_nid)
  local collective_type = std.as_read(collective.value.expr_type)
  local value_type = collective_type.result_type

  local stride = make_constant(shard_size, int, collective.value.span)
  local stride_minus_1 = make_constant(shard_size - 1, int, collective.value.span)

  local nparts_label = cx.graph:node_label(nparts_nid)
  local create_label = flow.node.Opaque {
    action = ast.typed.stat.Var {
      symbols = terralib.newlist({collective.value.value}),
      types = terralib.newlist({collective_type}),
      values = terralib.newlist({
          ast.typed.expr.DynamicCollective {
            arrivals = ast.typed.expr.Binary {
              lhs = ast.typed.expr.Binary {
                lhs = nparts_label.value,
                rhs = stride_minus_1,
                op = "+",
                expr_type = int,
                annotations = ast.default_annotations(),
                span = collective.value.span,
              },
              rhs = stride,
              op = "/",
              expr_type = int,
              annotations = ast.default_annotations(),
              span = collective.value.span,
            },
            op = op,
            value_type = value_type,
            expr_type = collective_type,
            annotations = ast.default_annotations(),
            span = collective.value.span,
          },
      }),
      annotations = ast.default_annotations(),
      span = collective.value.span,
    }
  }
  local create_nid = cx.graph:add_node(create_label)

  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    nparts_nid, cx.graph:node_result_port(nparts_nid),
    create_nid, cx.graph:node_available_port(create_nid))

  cx.graph:add_edge(
    flow.edge.HappensBefore {}, create_nid, cx.graph:node_sync_port(create_nid),
    collective_nid, cx.graph:node_sync_port(collective_nid))
end

local function issue_single_barrier_creation(cx, barrier_nid, nparts_nid)
  local barrier = cx.graph:node_label(barrier_nid)
  local barrier_type = std.as_read(barrier.value.expr_type)

  local stride = make_constant(shard_size, int, barrier.value.span)
  local stride_minus_1 = make_constant(shard_size - 1, int, barrier.value.span)

  local nparts_label = cx.graph:node_label(nparts_nid)
  local create_label = flow.node.Opaque {
    action = ast.typed.stat.Var {
      symbols = terralib.newlist({barrier.value.value}),
      types = terralib.newlist({barrier_type}),
      values = terralib.newlist({
          ast.typed.expr.PhaseBarrier {
            value = ast.typed.expr.Binary {
              lhs = ast.typed.expr.Binary {
                lhs = nparts_label.value,
                rhs = stride_minus_1,
                op = "+",
                expr_type = int,
                annotations = ast.default_annotations(),
                span = barrier.value.span,
              },
              rhs = stride,
              op = "/",
              expr_type = int,
              annotations = ast.default_annotations(),
              span = barrier.value.span,
            },
            expr_type = barrier_type,
            annotations = ast.default_annotations(),
            span = barrier.value.span,
          },
      }),
      annotations = ast.default_annotations(),
      span = barrier.value.span,
    }
  }
  local create_nid = cx.graph:add_node(create_label)
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    nparts_nid, cx.graph:node_result_port(nparts_nid),
    create_nid, cx.graph:node_available_port(create_nid))
  cx.graph:add_edge(
    flow.edge.HappensBefore {}, create_nid, cx.graph:node_sync_port(create_nid),
    barrier_nid, cx.graph:node_sync_port(barrier_nid))
end

local function issue_global_vars_creation_forlist(cx, loop_nid, nparts_nid,
                                                  part_indices_nid,
                                                  bound_label)

  local bound_nid = find_matching_input(cx, loop_nid,
    bound_label.region_type, bound_label.field_path)

  -- `var nparts = bound.volume`.
  local nparts_label = cx.graph:node_label(nparts_nid)
  local span = nparts_label.value.span
  local compute_nparts_nid = cx.graph:add_node(flow.node.Opaque {
    action = ast.typed.stat.Var {
      symbols = terralib.newlist({nparts_label.value.value}),
      types = terralib.newlist({int}),
      values = terralib.newlist({
        ast.typed.expr.FieldAccess {
          value = bound_label.value,
          field_name = "volume",
          expr_type = int,
          annotations = ast.default_annotations(),
          span = span,
        },
      }),
      annotations = ast.default_annotations(),
      span = span,
    }
  })
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    bound_nid, cx.graph:node_result_port(bound_nid),
    compute_nparts_nid, cx.graph:node_available_port(compute_nparts_nid))
  cx.graph:add_edge(
    flow.edge.HappensBefore {},
    compute_nparts_nid, cx.graph:node_sync_port(compute_nparts_nid),
    nparts_nid, cx.graph:node_sync_port(nparts_nid))

  -- `var part_indices = list_ispace(bound)`
  local part_indices_label = cx.graph:node_label(part_indices_nid)
  local part_indices_type = std.as_read(part_indices_label.value.expr_type)
  local compute_part_indices_nid = cx.graph:add_node(flow.node.Opaque {
    action = ast.typed.stat.Var {
      symbols = terralib.newlist({part_indices_label.value.value}),
      types = terralib.newlist({part_indices_type}),
      values = terralib.newlist({
        ast.typed.expr.ListIspace {
          ispace = bound_label.value,
          mapping = spmd_mapping.fn,
          key_type = spmd_mapping.key_type,
          expr_type = part_indices_type,
          annotations = ast.default_annotations(),
          span = span,
        },
      }),
      annotations = ast.default_annotations(),
      span = span,
    }
  })
  cx.graph:add_edge(
    flow.edge.Read(flow.default_mode()),
    bound_nid, cx.graph:node_result_port(bound_nid),
    compute_part_indices_nid, cx.graph:node_available_port(compute_part_indices_nid))
  cx.graph:add_edge(
    flow.edge.HappensBefore {},
    compute_part_indices_nid, cx.graph:node_sync_port(compute_part_indices_nid),
    part_indices_nid, cx.graph:node_sync_port(part_indices_nid))
end

local function issue_global_vars_creation_fornum(cx, loop_nid, nparts_nid,
                                                 part_indices_nid,
                                                 original_bounds)
  -- `var nparts = orig_bound2 - orig_bound1`.
  local nparts_label = cx.graph:node_label(nparts_nid)
  local span = nparts_label.value.span
  local nparts_type = std.as_read(nparts_label.value.expr_type)
  local compute_nparts_nid = cx.graph:add_node(flow.node.Opaque {
    action = ast.typed.stat.Var {
      symbols = terralib.newlist({nparts_label.value.value}),
      types = terralib.newlist({nparts_type}),
      values = terralib.newlist({
        ast.typed.expr.Binary {
          lhs = original_bounds[2].value,
          rhs = original_bounds[1].value,
          op = "-",
          expr_type = nparts_type,
          annotations = ast.default_annotations(),
          span = span,
        },
      }),
      annotations = ast.default_annotations(),
      span = span,
    }
  })
  cx.graph:add_edge(
    flow.edge.HappensBefore {},
    compute_nparts_nid, cx.graph:node_sync_port(compute_nparts_nid),
    nparts_nid, cx.graph:node_sync_port(nparts_nid))

  -- `var part_indices = list_range(orig_bound1, orig_bound2)`.
  local part_indices_label = cx.graph:node_label(part_indices_nid)
  local part_indices_type = std.as_read(part_indices_label.value.expr_type)
  local compute_part_indices_nid = cx.graph:add_node(flow.node.Opaque {
    action = ast.typed.stat.Var {
      symbols = terralib.newlist({part_indices_label.value.value}),
      types = terralib.newlist({part_indices_type}),
      values = terralib.newlist({
        ast.typed.expr.ListRange {
          start = original_bounds[1].value,
          stop = original_bounds[2].value,
          expr_type = part_indices_type,
          annotations = ast.default_annotations(),
          span = span,
        },
      }),
      annotations = ast.default_annotations(),
      span = span,
    }
  })
  cx.graph:add_edge(
    flow.edge.HappensBefore {},
    compute_part_indices_nid, cx.graph:node_sync_port(compute_part_indices_nid),
    part_indices_nid, cx.graph:node_sync_port(part_indices_nid))

  -- Wire original bounds as inputs to the new nodes.
  for _, bound_label in ipairs(original_bounds) do
    if bound_label:is(flow.node.data) then
      local bound_nid = find_matching_input(cx, loop_nid,
        bound_label.region_type, bound_label.field_path)
      cx.graph:add_edge(
        flow.edge.Read(flow.default_mode()), bound_nid, cx.graph:node_result_port(bound_nid),
        compute_nparts_nid, cx.graph:node_available_port(compute_nparts_nid))
      cx.graph:add_edge(
        flow.edge.Read(flow.default_mode()), bound_nid, cx.graph:node_result_port(bound_nid),
        compute_part_indices_nid, cx.graph:node_available_port(compute_part_indices_nid))
    end
  end
end

local function issue_global_vars_creation(cx, loop_nid, nparts_nid,
                                          part_indices_nid, original_bounds)

  if #original_bounds == 1 then -- ForList loop.
    issue_global_vars_creation_forlist(cx, loop_nid, nparts_nid,
      part_indices_nid, original_bounds[1])
  elseif #original_bounds == 2 then -- ForNum loop.
    issue_global_vars_creation_fornum(cx, loop_nid, nparts_nid,
      part_indices_nid, original_bounds)
  else
    assert(false)
  end
end

local function merge_open_nids(cx)
  local function is_read_to_open(edge)
    return is_read(edge) and
      cx.graph:node_label(edge.to_node):is(flow.node.Open)
  end
  local function is_bad(nid, label)
    return label:is(flow.node.data) and
      #(data.filter(is_read_to_open, cx.graph:outgoing_edges(nid))) > 1
  end

  local bad_nids = cx.graph:filter_nodes(is_bad)
  for _, bad_nid in ipairs(bad_nids) do
    local open_nid
    local outputs = cx.graph:outgoing_edges(bad_nid)
    for _, edge in ipairs(outputs) do
      local to_label = cx.graph:node_label(edge.to_node)
      if to_label:is(flow.node.Open) then
        if not open_nid then
          open_nid = edge.to_node
        else
          local open_outputs = cx.graph:outgoing_edges(edge.to_node)
          for _, edge in ipairs(open_outputs) do
            cx.graph:add_edge(
              edge.label,
              open_nid, cx.graph:node_result_port(open_nid),
              edge.to_node, edge.to_port)
          end
          cx.graph:remove_node(edge.to_node)
        end
      end
    end
  end
end

local function merge_close_nids(cx)
  local function is_write_from_close(edge)
    return is_write(edge) and
      cx.graph:node_label(edge.from_node):is(flow.node.Close)
  end
  local function is_bad(nid, label)
    return label:is(flow.node.data) and
      #(data.filter(is_write_from_close, cx.graph:incoming_edges(nid))) > 1
  end

  local bad_nids = cx.graph:filter_nodes(is_bad)
  for _, bad_nid in ipairs(bad_nids) do
    local close_nid
    local inputs = cx.graph:incoming_edges(bad_nid)
    for _, edge in ipairs(inputs) do
      local from_label = cx.graph:node_label(edge.from_node)
      if from_label:is(flow.node.Close) then
        if not close_nid then
          close_nid = edge.from_node
        else
          local close_inputs = cx.graph:incoming_edges(edge.from_node)
          for _, edge in ipairs(close_inputs) do
            cx.graph:add_edge(
              edge.label, edge.from_node, edge.from_port,
              close_nid, cx.graph:node_available_port(close_nid))
          end
          cx.graph:remove_node(edge.from_node)
        end
      end
    end
  end
end

local function rewrite_inputs(cx, old_loop, new_loop,
                              original_partitions, original_bounds,
                              elide_copy, shadowed_partitions, intersections, barriers, collectives,
                              start_barrier, global_vars, mapping)
  --  1. Find mapping from old to new inputs (outputs).
  --  2. For each input (output), either:
  --      a. Replace new with old (if they are identical).
  --      b. Add logic to duplicate/slice the input (for lists).
  --          i. Insert copies and opens/closes to make data consistent.
  --  3. Merge open and close nodes for racing copies.
  --  4. Copy happens-before edges for the loop node itself.

  -- Make global variables `nparts` and `part_indices`.
  local nparts_nid = find_matching_input(cx, new_loop,
    global_vars.nparts.region_type, global_vars.nparts.field_path)
  local part_indices_nid = find_matching_input(cx, new_loop,
    global_vars.part_indices.region_type, global_vars.part_indices.field_path)
  issue_global_vars_creation(cx, old_loop, nparts_nid, part_indices_nid, original_bounds)

  -- Compute more useful indexes for intersections and phase barriers.
  local partitions = data.new_recursive_map(1)
  for old_type, labels in original_partitions:items() do
    if mapping[old_type] then
      partitions[mapping[old_type]] = labels
    end
  end

  local intersection_types = data.newmap()
  for lhs_type, i1 in intersections:items() do
    for rhs_type, i2 in i1:items() do
      for field_path, intersection_label in i2:items() do
        intersection_types[intersection_label.region_type] = data.newtuple(
          lhs_type, rhs_type)
      end
    end
  end

  local barriers_empty_in = data.newmap()
  local barriers_empty_out = data.newmap()
  local barriers_full_in = data.newmap()
  local barriers_full_out = data.newmap()
  for lhs_type, b1 in barriers:items() do
    for rhs_type, b2 in b1:items() do
      for field_path, barrier_labels in b2:items() do
        local empty_in, empty_out, full_in, full_out = unpack(barrier_labels)
        barriers_empty_in[empty_in.region_type] = data.newtuple(
          lhs_type, rhs_type, field_path)
        barriers_empty_out[empty_out.region_type] = data.newtuple(
          lhs_type, rhs_type, field_path)
        barriers_full_in[full_in.region_type] = data.newtuple(
          lhs_type, rhs_type, field_path)
        barriers_full_out[full_out.region_type] = data.newtuple(
          lhs_type, rhs_type, field_path)
      end
    end
  end

  local collective_types = data.newmap()
  for _, c in ipairs(collectives) do
    local collective, op = unpack(c)
    collective_types[collective.region_type] = true
  end

  local inverse_mapping = {}
  for k, v in pairs(mapping) do
    if v then inverse_mapping[v] = k end
  end

  -- Find mapping from old to new inputs.
  local input_nid_mapping, output_nid_mapping = find_nid_mapping(
    cx, old_loop, new_loop, intersection_types,
    barriers_empty_in, barriers_empty_out, barriers_full_in, barriers_full_out,
    collective_types, start_barrier, global_vars, mapping)

  -- Rewrite inputs.
  local opened_nids = data.new_recursive_map(1)
  for region_type, region_fields in input_nid_mapping:items() do
    local need_copy = data.new_recursive_map(1)
    for field_path, nid_mapping in region_fields:items() do
      for old_nid, new_nid in nid_mapping:items() do
        local old_label = cx.graph:node_label(old_nid)
        local new_label = cx.graph:node_label(new_nid)
        if old_label:type() == new_label:type() then
          cx.graph:replace_node(new_nid, old_nid)
        elseif (old_label:is(flow.node.data.Region) or
                  old_label:is(flow.node.data.Partition) or
                  old_label:is(flow.node.data.CrossProduct)) and
          new_label:is(flow.node.data.List)
        then
          need_copy[field_path][old_nid] = new_nid
        else
          assert(false)
        end
      end
    end
    if not need_copy:is_empty() then
      local elide = elide_copy[region_type.partition_type]
      local shadow = shadowed_partitions[region_type.partition_type]
      local shadow_label = shadow and partitions[mapping[shadow]]
      if shadow then
        local has_intersection = false
        for _, pair in intersection_types:items() do
          local _, rhs_type = unpack(pair)
          if std.type_eq(rhs_type, region_type) then
            has_intersection = true
            break
          end
        end
        shadow = has_intersection and shadow
        shadow_label = shadow and shadow_label
      end
      issue_input_copies(
        cx, region_type, need_copy, elide, shadow_label, partitions,
        old_loop, nparts_nid, part_indices_nid, opened_nids)
    end
  end

  -- Rewrite outputs.
  for region_type, region_fields in output_nid_mapping:items() do
    local need_copy = data.new_recursive_map(1)
    for field_path, nid_mapping in region_fields:items() do
      for old_nid, new_nid in nid_mapping:items() do
        local old_label = cx.graph:node_label(old_nid)
        local new_label = cx.graph:node_label(new_nid)
        if old_label:type() == new_label:type() then
          cx.graph:replace_node(new_nid, old_nid)
        elseif (old_label:is(flow.node.data.Region) or
                  old_label:is(flow.node.data.Partition) or
                  old_label:is(flow.node.data.CrossProduct)) and
          new_label:is(flow.node.data.List)
        then
          need_copy[field_path][old_nid] = new_nid
        else
          assert(false)
        end
      end
    end
    if not need_copy:is_empty() then
      local elide = elide_copy[region_type.partition_type]
      local shadow = shadowed_partitions[region_type.partition_type]
      local shadow_label = shadow and partitions[mapping[shadow]]
      issue_output_copies(
        cx, region_type, need_copy, elide, shadow_label, partitions,
        old_loop, nparts_nid, part_indices_nid, opened_nids)
    end
  end

  -- Rewrite intersections.
  for intersection_type, list_types in intersection_types:items() do
    local lhs_type, rhs_type = unpack(list_types)
    local intersection_nids = find_matching_inputs(
      cx, new_loop, intersection_type)
    local lhs_nid = find_matching_input(cx, new_loop, lhs_type)
    local rhs_nid = find_matching_input(cx, new_loop, rhs_type)
    assert(#intersection_nids > 0 and lhs_nid and rhs_nid)
    issue_intersection_creation(cx, intersection_nids, lhs_nid, rhs_nid)
  end

  -- Rewrite barriers.
  for lhs_type, b1 in barriers:items() do
    for rhs_type, b2 in b1:items() do
      for field_path, barrier_labels in b2:items() do
        local empty_in, empty_out, full_in, full_out = unpack(barrier_labels)

        local rhs_nid = find_matching_input(cx, new_loop, rhs_type, field_path)

        local intersection_type = intersections[lhs_type][rhs_type][field_path].region_type
        local intersection_nid = find_matching_input(
          cx, new_loop, intersection_type, field_path)

        local empty_in_nid = find_matching_input(
          cx, new_loop, empty_in.region_type)
        local empty_out_nid = find_matching_input(
          cx, new_loop, empty_out.region_type)
        local full_in_nid = find_matching_input(
          cx, new_loop, full_in.region_type)
        local full_out_nid = find_matching_input(
          cx, new_loop, full_out.region_type)

        assert(intersection_nid and empty_in_nid and empty_out_nid and
                 full_in_nid and full_out_nid)

        issue_barrier_creation(
          cx, rhs_nid, intersection_nid, empty_in_nid, empty_out_nid)
        issue_barrier_creation(
          cx, rhs_nid, intersection_nid, full_in_nid, full_out_nid)
      end
    end
  end

  -- Rewrite collectives.
  for _, c in ipairs(collectives) do
    local collective, op = unpack(c)
    local collective_nid = find_matching_input(
      cx, new_loop, collective.region_type, collective.field_path)
    issue_collective_creation(cx, old_loop, collective_nid, op, nparts_nid)
  end

  local start_barrier_nid = find_matching_input(
    cx, new_loop, start_barrier.region_type, start_barrier.field_path)
  issue_single_barrier_creation(cx, start_barrier_nid, nparts_nid)

  -- Merge open and close nodes for racing copies.

  -- Sometimes this algorith generates copies which have the potential
  -- to race. This is actually ok, because the data being copied is
  -- the same (as guarranteed by the interior of the task). However,
  -- the dataflow code generator will complain if the copies are
  -- visibly racing. Therefore, merge the open and close operations to
  -- avoid the appearance of a race.
  merge_open_nids(cx)
  merge_close_nids(cx)

  -- Copy happens-before edges for the loop node itself.
  cx.graph:copy_incoming_edges(is_happens_before, old_loop, new_loop)
  cx.graph:copy_outgoing_edges(is_happens_before, old_loop, new_loop)
end

local function compose_mapping(old, new)
  local result = {}
  for k, v in pairs(new) do
    result[k] = v
  end
  for k, v in pairs(old) do
    result[k] = (new[v] ~= nil and new[v]) or v
  end
  return result
end

local function spmdize(cx, loop)
  --  1. Extract shard (deep copy).
  --  2. Normalize communication graph (remove opens).
  --  3. Rewrite shard partitions as lists.
  --  4. Rewrite communication graph (change closes to copies).
  --  5. Rewrite reductions to use temporary scratch fields.
  --  6. Rewrite shard loop bounds.
  --  7. Upgrade everything to simultaneous coherence (and no_access_flag).
  --  8. Outline shard into a task.
  --  9. Compute shard bounds and list slices.
  -- 10. Wrap that in a distribution loop.
  -- 11. Downgrade everything back to exclusive.
  -- 12. Wrap that in a must epoch.
  -- 13. Outline that into a task.
  -- 14. Rewrite inputs/outputs.

  local span = cx.graph:node_label(loop).span
  local annotations = cx.graph:node_label(loop).annotations
  local leaf_loop_type = find_leaf_loop_type(cx, loop)

  local shard_graph, shard_loop = flow_extract_subgraph.entry(cx.graph, loop)

  local shard_cx = cx:new_graph_scope(shard_graph)
  normalize_communication(shard_cx)
  local lists, original_partitions, mapping = rewrite_shard_partitions(shard_cx)
  -- Compute copy elision early to avoid copy-ins inside the shard
  local elide_copy, shadowed_partitions = select_copy_elision(shard_cx, lists, mapping)
  local original_intersections, barriers = rewrite_communication(shard_cx, shard_loop, mapping)
  local collectives, local_collectives = rewrite_scalar_communication(shard_cx)
  local scratch_field_mapping = rewrite_reduction_scratch_fields(shard_cx, shard_loop)
  local intersections, intersection_mapping, intersection_preconditions = rewrite_shard_intersections(
    shard_cx, shard_loop, original_intersections)
  local copyin_preconditions, copyin_mapping = rewrite_initial_copyin(
    shard_cx, shard_loop, original_intersections, intersection_preconditions, elide_copy, mapping)

  -- `shard_vars` contains labels for variables specific to a shard
  -- (e.g. the range of global indices it is responsible for).
  local shard_vars, original_bounds
  if leaf_loop_type:is(flow.node.ctrl.ForNum) then
    shard_vars, original_bounds = rewrite_fornum_loops_in_shard(shard_cx, shard_loop)
  elseif leaf_loop_type:is(flow.node.ctrl.ForList) then
    shard_vars, original_bounds = rewrite_forlist_loops_in_shard(shard_cx, shard_loop)
  else
    assert(false)
  end

  -- `global_vars` contains labels for variables pertaining to the entire SPMD block
  -- (e.g. the list of all global indices).  Their creation is issued within
  -- `rewrite_inputs`.
  local global_vars = make_global_vars(cx, shard_vars, span)
  local collective_mapping = rewrite_local_collective_creation(
    shard_cx, shard_loop, shard_vars, local_collectives)
  local start_barrier = synchronize_shard_start(shard_cx, shard_loop, copyin_preconditions)

  local shard_mapping = compose_mapping(
    compose_mapping(
      compose_mapping(collective_mapping, scratch_field_mapping),
      intersection_mapping),
    copyin_mapping)
  local outer_shard_cx = cx:new_graph_scope(flow.empty_graph(cx.tree))
  local outer_shard_block = make_block(
    outer_shard_cx, shard_cx.graph, shard_mapping, span)
  upgrade_simultaneous_coherence(outer_shard_cx)

  local shard_task = flow_outline_task.entry(
    outer_shard_cx.graph, outer_shard_block, "shard", true)
  local elided_lists, elided_intersections, elided_barriers, elided_mapping =
    rewrite_list_elision(outer_shard_cx, lists, intersections, barriers, mapping, elide_copy, shadowed_partitions)
  local shard_index, shard_stride, slice_mapping,
      new_intersections, new_barriers = rewrite_shard_slices(
      outer_shard_cx, shard_vars, global_vars, elided_lists, elided_intersections,
      elided_barriers, shard_task, compose_mapping(mapping, elided_mapping))

  local dist_cx = cx:new_graph_scope(flow.empty_graph(cx.tree))
  local dist_loop = make_distribution_loop(
    dist_cx, outer_shard_cx.graph, shard_index, shard_stride, global_vars,
    compose_mapping(elided_mapping, slice_mapping), span)
  downgrade_simultaneous_coherence(dist_cx)

  local epoch_loop = make_must_epoch(cx, dist_cx.graph, annotations, span)

  local inputs_mapping = compose_mapping(
    mapping, compose_mapping(elided_mapping, slice_mapping))

  rewrite_inputs(cx, loop, epoch_loop, original_partitions, original_bounds,
                 elide_copy, shadowed_partitions, new_intersections, new_barriers, collectives,
                 start_barrier, global_vars, inputs_mapping)

  return epoch_loop
end

local function spmdize_eligible_loop(cx, loops)
  for loop, _ in pairs(loops) do
    if can_spmdize(cx, loop) then
      local new_loop = spmdize(cx, loop)
      cx.graph:remove_node(loop)
      loops[loop] = nil
      -- Don't need to re-analyze already-SPMDized loops.
      -- loops[new_loop] = true
      return true
    end
  end
  return false
end

local function spmdize_eligible_loops(cx, original_loops)
  local loops = {}
  for _, nid in pairs(original_loops) do
    loops[nid] = true
  end
  repeat until not spmdize_eligible_loop(cx, loops)
end

function flow_spmd.graph(cx, graph)
  assert(flow.is_graph(graph))
  local cx = cx:new_graph_scope(graph)
  local loops = cx.graph:filter_nodes(
    function(_, label) return label:is(flow.node.ctrl) end)
  spmdize_eligible_loops(cx, loops)
  return cx.graph
end

function flow_spmd.top_task(cx, node)
  if not node.body then return node end

  local body = node.body:deepcopy()
  body:map_nodes_recursive(
    function(graph, nid, label)
      if label:is(flow.node.ctrl) then
        return label { block = flow_spmd.graph(cx, label.block) }
      end
      return label
    end)
  body = flow_spmd.graph(cx, body)
  return node { body = body }
end

function flow_spmd.top(cx, node)
  if node:is(ast.typed.top.Task) then
    return flow_spmd.top_task(cx, node)

  else
    return node
  end
end

function flow_spmd.entry(node)
  local cx = context.new_global_scope()
  return flow_spmd.top(cx, node)
end

flow_spmd.pass_name = "flow_spmd"

if std.config["flow"] and std.config["flow-spmd"] then passes_hooks.add_optimization(16, flow_spmd) end

return flow_spmd

