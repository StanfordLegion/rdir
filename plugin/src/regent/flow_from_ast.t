-- Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
-- Copyright (c) 2015-2018, Stanford University. All rights reserved.
--
-- This file was initially released under the BSD license, shown
-- below. All subsequent contributions are dual-licensed under the BSD
-- and Apache version 2.0 licenses.
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

-- Conversion from AST to Dataflow IR

local ast = require("regent/ast")
local data = require("common/data")
local flow = require("regent/flow")
local flow_region_tree = require("regent/flow_region_tree")
local passes_hooks = require("regent/passes_hooks")
local std = require("regent/std")
local symbol_table = require("regent/symbol_table")

-- Field Maps

local _field_map = {}
_field_map.__index = _field_map

function is_field_map(x)
  return getmetatable(x) == _field_map
end

function new_field_map()
  return setmetatable({ keys = {}, values = {} }, _field_map)
end

local function _field_map_next_key(t, i)
  local ih
  if i ~= nil then
    ih = i:hash()
  end
  local kh, k = next(t.keys, ih)
  return k, k and t.values[kh]
end

function _field_map:items()
  return _field_map_next_key, self, nil
end

function _field_map:map(f)
  local result = new_field_map()
  for k, v in self:items() do
    result:insert(k, f(k, v))
  end
  return result
end

function _field_map:maplist(f)
  local result = terralib.newlist()
  for k, v in self:items() do
    result:insert(f(k, v))
  end
  return result
end

function _field_map:prepend(p)
  if type(p) == "string" then
    p = data.newtuple(p)
  end
  assert(data.is_tuple(p))
  local result = new_field_map()
  for k, v in self:items() do
    result:insert(p .. k, v)
  end
  return result
end

function _field_map:contains(k)
  return self.values[k:hash()]
end

function _field_map:is_empty()
  for k, v in self:items() do
    return false
  end
  return true
end

function _field_map:lookup(k)
  local v = self.values[k:hash()]
  if v == nil then
    error("field map has no such key " .. tostring(k))
  end
  return v
end

function _field_map:insert(k, v)
  assert(data.is_tuple(k))
  local kh = k:hash()
  self.keys[kh] = k
  self.values[kh] = v
end

function _field_map:insertall(t)
  assert(is_field_map(t))
  for k, v in t:items() do
    self:insert(k, v)
  end
end

function _field_map:__tostring()
  return "{" .. self:hash() .. "}"
end

local function _field_map_tostring(k, v)
  return tostring(k) .. " = " .. tostring(v)
end

function _field_map:concat(sep)
  return self:maplist(_field_map_tostring):concat(", ")
end

function _field_map:hash()
  return self:concat(",")
end

-- Privileges

-- FIXME: We're going to use physical privileges here. TBD whether
-- this could be cleaned up to use normal Regent privileges. For now
-- just declare the necessary helpers.

local function reduces(op)
  return tostring(std.reduces(op))
end

-- Context

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

local region_tree_state

function context:new_local_scope(local_var)
  local local_vars = self.local_vars:copy()
  if local_var then
    local_vars[local_var] = true
  end
  local cx = {
    constraints = self.constraints,
    graph = flow.empty_graph(self.tree),
    local_vars = local_vars,
    epoch = terralib.newlist(),
    next_epoch = terralib.newlist(),
    next_epoch_opaque = false,
    tree = self.tree,
    region_symbols = self.region_symbols:new_local_scope(),
    state_by_field = new_field_map(),
  }
  return setmetatable(cx, context)
end

function context:new_task_scope(constraints, region_universe)
  local tree = flow_region_tree.new_region_tree(constraints, region_universe)
  local cx = {
    constraints = constraints,
    graph = flow.empty_graph(tree),
    local_vars = data.newmap(),
    epoch = terralib.newlist(),
    next_epoch = terralib.newlist(),
    next_epoch_opaque = false,
    tree = tree,
    region_symbols = self.region_symbols:new_local_scope(),
    state_by_field = new_field_map(),
  }
  return setmetatable(cx, context)
end

function context.new_global_scope()
  local cx = {
    region_symbols = symbol_table.new_global_scope({}),
  }
  return setmetatable(cx, context)
end

function context:intern_region(node, var_symbol, var_type)
  local region_type = self.tree:intern_variable(var_type, var_symbol, node.annotations, node.span)
  self.region_symbols:insert(node, region_type, var_symbol)

  local value_type = std.as_read(var_type)
  if std.is_region(value_type) and not self.tree:is_point(region_type) then
    local partition = self.tree:point_partition(region_type)
    self:intern_region(node, std.newsymbol(partition), partition)
  end
  if std.is_partition(value_type) or std.is_cross_product(value_type) then
    local parent = value_type:parent_region()
    if not self:has_region_symbol(parent) then
      self:intern_region(node, std.newsymbol(parent), parent)
    end
  end
  if std.is_cross_product(value_type) then
    local partition = value_type:partition()
    if not self:has_region_symbol(partition) then
      self:intern_region(node, std.newsymbol(partition), partition)
    end
  end
end

function context:intern_region_point_expr(node, var_symbol, region_type)
  assert(std.is_symbol(var_symbol) and std.is_region(region_type))
  assert(self.tree:is_point(region_type))

  self.region_symbols:insert(node, region_type, var_symbol)
end

function context:has_region_symbol(region_type)
  assert(flow_region_tree.is_region(region_type))

  return self.region_symbols:safe_lookup(region_type)
end

function context:region_symbol(region_type)
  assert(flow_region_tree.is_region(region_type))

  local symbol = self.region_symbols:safe_lookup(region_type)
  if not symbol then
    print("undefined region", region_type, self.tree:is_point(region_type))
    if std.is_partition(region_type) then
      print(" (point partition?)", region_type:parent_region(), self.tree:point_partition(region_type:parent_region()))
    end
  end
  assert(symbol)
  return symbol
end

function context:state(field_path)
  assert(data.is_tuple(field_path))
  if self.state_by_field:contains(field_path) then
    return self.state_by_field:lookup(field_path)
  end

  local state = region_tree_state.new(self.tree)
  self.state_by_field:insert(field_path, state)
  return state
end

-- Graph Construction

local function as_nid(cx, value)
  local nid
  for field_path, values in value:items() do
    local privilege, input_nid, output_nid = unpack(values)
    if nid == nil then
      nid = input_nid or output_nid
      break
    end
  end
  assert(nid)
  return nid
end

local function as_ast(cx, value)
  local nid = as_nid(cx, value)
  return cx.graph:node_label(nid).value
end

local function sequence_depend(cx, nid)
  local label = cx.graph:node_label(nid)
  local opaque = flow.is_opaque_node(label)
  if opaque and not cx.next_epoch_opaque then
    if #cx.next_epoch > 0 then
      cx.epoch = cx.next_epoch
      cx.next_epoch = terralib.newlist()
    end
    cx.next_epoch_opaque = true
  end
  for _, epoch_nid in ipairs(cx.epoch) do
    if not cx.graph:reachable(epoch_nid, nid) then
      cx.graph:add_edge(
        flow.edge.HappensBefore {},
        epoch_nid, cx.graph:node_sync_port(epoch_nid),
        nid, cx.graph:node_sync_port(nid))
    end
  end
  return nid
end

local function sequence_advance(cx, nid)
  cx.next_epoch:insert(nid)
  if cx.next_epoch_opaque then
    cx.epoch = cx.next_epoch
    cx.next_epoch = terralib.newlist()
    cx.next_epoch_opaque = false
  end
  return nid
end

local function add_node(cx, label)
  return cx.graph:add_node(label)
end

local function add_input_edge(cx, from_nid, to_nid, to_port, privilege)
  assert(to_port > 0)
  local label
  if privilege == "none" then
    label = flow.edge.None(flow.default_mode())
  elseif privilege == "reads" or privilege == "reads_writes" then
    label = flow.edge.Read(flow.default_mode())
  else
    assert(false)
  end
  cx.graph:add_edge(
    label,
    from_nid, cx.graph:node_result_port(from_nid),
    to_nid, to_port)
end

local function add_output_edge(cx, from_nid, from_port, to_nid, privilege)
  assert(from_port > 0)
  local label
  if privilege == "reads_writes" then
    label = flow.edge.Write(flow.default_mode())
  elseif std.is_reduction_op(privilege) then
    label = flow.edge.Reduce {
      coherence = flow.default_coherence(),
      flag = flow.default_flag(),
      op = std.get_reduction_op(privilege)
    }
  else
    assert(false)
  end
  cx.graph:add_edge(label, from_nid, from_port, to_nid, 0)
end

local function add_name_edge(cx, from_nid, to_nid)
  cx.graph:add_edge(
    flow.edge.Name {},
    from_nid, cx.graph:node_result_port(from_nid),
    to_nid, 0)
end

local function add_args(cx, compute_nid, args)
  for i, arg in pairs(args) do
    assert(is_field_map(arg))
    for field_path, values in arg:items() do
      local privilege, input_nid, output_nid = unpack(values)
      if input_nid then
        add_input_edge(cx, input_nid, compute_nid, i, privilege)
      end
      if output_nid then
        add_output_edge(cx, compute_nid, i, output_nid, privilege)
      end
    end
  end
end

local function add_result(cx, from_nid, expr_type, annotations, span)
  if expr_type == terralib.types.unit then
    return from_nid
  end

  local symbol = std.newsymbol(expr_type)
  local region_type = cx.tree:intern_variable(expr_type, symbol, annotations, span)
  local label = ast.typed.expr.ID {
    value = symbol,
    expr_type = expr_type,
    annotations = annotations,
    span = span,
  }
  local result_nid = cx.graph:add_node(
    flow.node.data.Scalar {
      value = label,
      region_type = region_type,
      field_path = data.newtuple(),
      fresh = true,
  })
  local edge_label
  if flow_region_tree.is_region(expr_type) then
    edge_label = flow.edge.Name {}
  else
    edge_label = flow.edge.Write(flow.default_mode())
  end
  cx.graph:add_edge(
    edge_label,
    from_nid, cx.graph:node_result_port(from_nid),
    result_nid, 0)
  return result_nid
end

-- Region Tree State

local region_state = setmetatable({}, { __index = function(t, k) error("region state has no field " .. tostring(k), 2) end})
region_state.__index = region_state

local modes = setmetatable({}, { __index = function(t, k) error("no such mode " .. tostring(k), 2) end})
modes.closed = "closed"
modes.read = "read"
modes.write = "write"
modes.reduce = "reduce"

local function is_mode(x)
  return rawget(modes, x)
end

region_tree_state = setmetatable({}, { __index = function(t, k) error("region tree state has no field " .. tostring(k), 2) end})
region_tree_state.__index = region_tree_state

function region_tree_state.new(tree)
  return setmetatable(
    {
      tree = tree,
      region_tree_state = {},
    }, region_tree_state)
end

function region_tree_state:ensure(region_type)
  assert(flow_region_tree.is_region(region_type))
  if not rawget(self.region_tree_state, region_type) then
    self.region_tree_state[region_type] = region_state.new()
  end
end

function region_tree_state:mode(region_type)
  assert(rawget(self.region_tree_state, region_type))
  return self.region_tree_state[region_type].mode
end

function region_tree_state:set_mode(region_type, mode)
  assert(rawget(self.region_tree_state, region_type))
  assert(is_mode(mode))
  self.region_tree_state[region_type].mode = mode
end

function region_tree_state:op(region_type)
  assert(rawget(self.region_tree_state, region_type))
  return self.region_tree_state[region_type].op
end

function region_tree_state:set_op(region_type, op)
  assert(rawget(self.region_tree_state, region_type))
  self.region_tree_state[region_type].op = op
end

function region_tree_state:current(region_type)
  assert(rawget(self.region_tree_state, region_type))
  return self.region_tree_state[region_type].current
end

function region_tree_state:set_current(region_type, nid)
  assert(rawget(self.region_tree_state, region_type))
  assert(flow.is_valid_node(nid))
  self.region_tree_state[region_type].current = nid
end

function region_tree_state:open(region_type)
  assert(rawget(self.region_tree_state, region_type))
  return self.region_tree_state[region_type].open
end

function region_tree_state:set_open(region_type, nid)
  assert(rawget(self.region_tree_state, region_type))
  assert(flow.is_null(nid) or flow.is_valid_node(nid))
  self.region_tree_state[region_type].open = nid
end

function region_tree_state:dirty(region_type)
  assert(rawget(self.region_tree_state, region_type))
  return self.region_tree_state[region_type].dirty
end

function region_tree_state:set_dirty(region_type, dirty)
  assert(rawget(self.region_tree_state, region_type))
  assert(type(dirty) == "boolean")
  self.region_tree_state[region_type].dirty = dirty
end

function region_tree_state:clear(region_type)
  assert(rawget(self.region_tree_state, region_type))
  self.region_tree_state[region_type] = region_state.new()
end

function region_tree_state:dirty_children(region_type)
  local result = terralib.newlist()
  for _, child in ipairs(self.tree:children(region_type)) do
    if self:dirty(child) then
      result:insert(child)
    end
  end
  return result
end

function region_tree_state:open_siblings(region_type)
  local result = terralib.newlist()
  for _, sibling in ipairs(self.tree:siblings(region_type)) do
    self:ensure(sibling)
    if self:mode(sibling) ~= modes.closed then
      result:insert(sibling)
    end
  end
  return result
end

function region_tree_state:current_siblings(region_type)
  local result = terralib.newlist()
  for _, sibling in ipairs(self.tree:siblings(region_type)) do
    self:ensure(sibling)
    if flow.is_valid_node(self:current(sibling)) then
      result:insert(sibling)
    end
  end
  return result
end

function region_tree_state:dirty_siblings(region_type)
  local result = terralib.newlist()
  for _, sibling in ipairs(self.tree:siblings(region_type)) do
    self:ensure(sibling)
    if self:dirty(sibling) then
      result:insert(sibling)
    end
  end
  return result
end

function region_state.new()
  return setmetatable({
      mode = modes.closed,
      current = flow.null(),
      open = flow.null(),
      dirty = false,
      op = false,
  }, region_state)
end

-- Region Identity Analysis

local analyze_regions = {}

function analyze_regions.vars(cx)
  return function(node)
    if node:is(ast.typed.stat.Var) then
      local var_type = std.rawref(&node.type)
      cx.tree:intern_variable(var_type, node.symbol, node.annotations, node.span)
    elseif node:is(ast.typed.stat.VarUnpack) then
      for i, var_symbol in ipairs(node.symbols) do
        local var_type = std.rawref(&node.field_types[i])
        cx.tree:intern_variable(var_type, var_symbol, node.annotations, node.span)
      end
    elseif node:is(ast.typed.stat.ForNum) or node:is(ast.typed.stat.ForList) then
      local var_symbol = node.symbol
      local var_type = node.symbol:gettype()
      cx.tree:intern_variable(var_type, var_symbol, node.annotations, node.span)
    elseif node:is(ast.typed.top.Task) then
      for i, param in ipairs(node.params) do
        local param_type = std.rawref(&param.param_type)
        cx.tree:intern_variable(
          param_type, param.symbol, param.annotations, param.span)
      end
    end
  end
end

function analyze_regions.expr(cx)
  return function(node)
    local expr_type = std.as_read(node.expr_type)
    if flow_region_tree.is_region(expr_type) then
      cx.tree:intern_region_expr(node.expr_type, node.annotations, node.span)
      if node:is(ast.typed.expr.IndexAccess) and
        not std.is_list_of_regions(std.as_read(node.value.expr_type))
      then
        cx.tree:attach_region_index(expr_type, node.index)
      end
    elseif std.is_bounded_type(expr_type) then
      -- This may have been the result of an unpack, in which case the
      -- regions in this bounded type may be fresh. Intern them just
      -- in case.
      for _, bound in ipairs(expr_type:bounds()) do
        if flow_region_tree.is_region(bound) then
          cx.tree:intern_region_expr(bound, node.annotations, node.span)
        end
      end
    end

    if node:is(ast.typed.expr.Deref) then
      local value_type = std.as_read(node.value.expr_type)
      if std.is_bounded_type(value_type) then
        local bounds = value_type:bounds()
        for _, parent in ipairs(bounds) do
          local index
          -- FIXME: This causes issues with some tests.
          -- if node.value:is(ast.typed.expr.ID) and
          --   not std.is_rawref(node.value.expr_type)
          -- then
          --   index = node.value
          -- end
          cx.tree:intern_region_point_expr(
            parent, index, node.annotations, node.span)
        end
      end
    end
  end
end

function analyze_regions.top_task(cx, node)
  ast.traverse_node_postorder(analyze_regions.vars(cx), node)
  ast.traverse_expr_postorder(analyze_regions.expr(cx), node)
end

-- Region Tree Analysis

local function expand_fields(region_type, privilege_map)
  local fspace = region_type:fspace()
  if not fspace:isstruct() or fspace == terralib.types.unit then
    return privilege_map
  end

  local all_fields = std.flatten_struct_fields(fspace)
  local result = new_field_map()
  for field_path, privilege in privilege_map:items() do
    if privilege == "none" then
      result:insert(field_path, privilege)
    else
      local fields = data.filter(
        function(field) return field:starts_with(field_path) end, all_fields)
      assert(#fields > 0)
      for _, field in ipairs(fields) do
        result:insert(field, privilege)
      end
    end
  end
  return result
end

local function get_expanded_fields(value_type, field_path)
  local field_paths, field_types = std.flatten_struct_fields(value_type)
  local result_fields, result_types = terralib.newlist(), terralib.newlist()
  for i, field in ipairs(field_paths) do
    local field_type = field_types[i]
    if field:starts_with(field_path) then
      result_fields:insert(field)
      result_types:insert(field_type)
    end
  end
  assert(#result_fields > 0)
  return result_fields, result_types
end

local function get_expanded_field(value_type, field_path)
  local field_paths, field_types = get_expanded_fields(value_type, field_path)
  assert(#field_paths == 1 and #field_types == 1)
  return field_types[1]
end

local function privilege_mode(privilege)
  if privilege == "none" then
    return false, false
  elseif privilege == "reads" then
    return modes.read, false
  elseif privilege == "reads_writes" then
    return modes.write, false
  elseif std.is_reduction_op(privilege) then
    return modes.reduce, std.get_reduction_op(privilege)
  else
    assert(false)
  end
end

local function get_region_label(cx, region_type, field_path)
  local symbol = cx:region_symbol(region_type)
  local expr_type = cx.tree:region_var_type(region_type)
  local name = ast.typed.expr.ID {
    value = cx:region_symbol(region_type),
    expr_type = expr_type,
    annotations = cx.tree:region_annotations(region_type),
    span = cx.tree:region_span(region_type),
  }
  if std.is_region(std.as_read(expr_type)) then
    if cx.tree:is_point(region_type) then
      -- FIXME: When we model assignments, this will need to become
      -- more expressive (i.e. for l-vals to work properly, this will
      -- need to be a deref, not the result of a deref).
      local parent = cx.tree:parent(cx.tree:parent(region_type))
      local expr_type = get_expanded_field(parent:fspace(), field_path)
      name = name {
        value = std.newsymbol(expr_type),
        expr_type = expr_type,
      }
    end
    return flow.node.data.Region {
      value = name,
      region_type = region_type,
      field_path = field_path,
    }
  elseif std.is_partition(std.as_read(expr_type)) then
    return flow.node.data.Partition {
      value = name,
      region_type = region_type,
      field_path = field_path,
    }
  elseif std.is_cross_product(std.as_read(expr_type)) then
    return flow.node.data.CrossProduct {
      value = name,
      region_type = region_type,
      field_path = field_path,
    }
  elseif std.is_list_of_regions(std.as_read(expr_type)) then
    return flow.node.data.List {
      value = name,
      region_type = region_type,
      field_path = field_path,
    }
  else
    assert(not flow_region_tree.is_region(std.as_read(expr_type)))
    return flow.node.data.Scalar {
      value = name,
      region_type = region_type,
      field_path = field_path,
      fresh = false,
    }
  end
end

local transitions = setmetatable(
  {}, { __index = function(t, k) error("no such transition " .. tostring(k), 2) end})

function transitions.nothing(cx, path, index, field_path)
  return cx:state(field_path):current(path[index]), false
end

function transitions.create(cx, path, index, field_path)
  local current_nid = cx:state(field_path):current(path[index])
  if not flow.is_null(current_nid) then
    return current_nid, false
  end

  local next_nid = cx.graph:add_node(get_region_label(cx, path[index], field_path))
  local parent_index = index + 1
  if parent_index <= #path then
    local open_nid = cx:state(field_path):open(path[parent_index])
    if flow.is_valid_node(open_nid) then
      add_output_edge(cx, open_nid, 1, next_nid, "reads_writes")
    end
  end
  cx:state(field_path):set_current(path[index], next_nid)
  return next_nid, true
end

function transitions.open(cx, path, index, field_path)
  local current_nid, fresh = transitions.create(cx, path, index, field_path)
  assert(flow.is_null(cx:state(field_path):open(path[index])))
  local open_nid = cx.graph:add_node(flow.node.Open {})
  add_input_edge(cx, current_nid, open_nid, 1, "reads")
  cx:state(field_path):set_open(path[index], open_nid)

  -- Add sequence dependencies here to avoid redundant edges already
  -- encoded by true data dependencies.
  if fresh then sequence_depend(cx, current_nid) end
end

function transitions.close(cx, path, index, field_path)
  -- Close all children.
  for _, child in ipairs(cx.tree:children(path[index])) do
    cx:state(field_path):ensure(child)
    if cx:state(field_path):mode(child) ~= modes.closed then
      local child_path = data.newtuple(child) .. path:slice(index, #path)
      local child_nid = cx:state(field_path):current(child)
      assert(flow.is_valid_node(child_nid))
      transitions.close(cx, child_path, 1, field_path)
    end
  end

  -- Create and link the close node.
  local close_nid = cx.graph:add_node(flow.node.Close {})
  add_input_edge(cx, cx:state(field_path):current(path[index]), close_nid, 1, "reads")
  local port = 2
  for _, child in ipairs(cx.tree:children(path[index])) do
    local child_nid = cx:state(field_path):current(child)
    if flow.is_valid_node(child_nid) then
      add_input_edge(cx, child_nid, close_nid, port, "reads")
      port = port + 1
    end
    cx:state(field_path):clear(child)
  end

  -- Create and link the next node.
  local next_nid = cx.graph:add_node(get_region_label(cx, path[index], field_path))
  add_output_edge(cx, close_nid, 1, next_nid, "reads_writes")

  -- Set node state.
  cx:state(field_path):set_mode(path[index], modes.closed)
  cx:state(field_path):set_current(path[index], next_nid)
  cx:state(field_path):set_open(path[index], flow.null())
  cx:state(field_path):set_dirty(path[index], true)

  return next_nid, true
end

function transitions.close_conflicting_children(cx, path, index, field_path)
  assert(false) -- FIXME: This code doesn't work.
  for _, child in ipairs(cx.tree:children(path[index])) do
    cx:state(field_path):ensure(child)
    if cx:state(field_path):mode(child) ~= modes.closed then
      local child_path = data.newtuple(child) .. path:slice(index, #path)
      transitions.close(
        cx, child_path, 1,
        cx.graph:node_label(cx:state(field_path):current(child)).value,
        field_path)
    end
  end
end

function transitions.close_and_reopen(cx, path, index, field_path)
  transitions.close(cx, path, index, field_path)
  transitions.open(cx, path, index, field_path)
end

local function select_transition(cx, path, index,
                                 desired_mode, desired_op, field_path)
  local current_mode = cx:state(field_path):mode(path[index])
  local current_op = cx:state(field_path):op(path[index])
  local current_nid = cx:state(field_path):current(path[index])
  if index == 1 then -- Leaf
    if current_mode == modes.closed then
      if desired_op ~= current_op and flow.is_valid_node(current_nid) then
        return modes.closed, desired_op, transitions.close
      else
        return modes.closed, desired_op, transitions.create
      end
    elseif current_mode == modes.read then
      if desired_mode == modes.read then
        return modes.read, desired_op, transitions.nothing
      else
        return modes.closed, desired_op, transitions.close
      end
    elseif current_mode == modes.write then
      return modes.closed, desired_op, transitions.close
    elseif current_mode == modes.reduce then
      if desired_mode == modes.reduce and desired_op == current_op then
        return modes.reduce, desired_op, transitions.nothing
      else
        return modes.closed, desired_op, transitions.close
      end
    else
      assert(false)
    end
  else -- Inner
    local child_index = index - 1
    if current_mode == modes.closed then
      return desired_mode, desired_op, transitions.open
    elseif current_mode == modes.read then
      if desired_mode == modes.read then
        return modes.read, desired_op, transitions.nothing
      else
        if desired_mode == modes.reduce or
          #cx:state(field_path):current_siblings(path[child_index]) > 0
        then
          return desired_mode, desired_op, transitions.close_and_reopen
        else
          return desired_mode, desired_op, transitions.nothing
        end
      end
    elseif current_mode == modes.write then
      -- FIXME: Does dirty include all open siblings?
      if #cx:state(field_path):dirty_siblings(path[child_index]) > 0 then
        return desired_mode, desired_op, transitions.close_and_reopen
      else
        return modes.write, false, transitions.nothing
      end
    elseif current_mode == modes.reduce then
      if desired_mode == modes.reduce then
        if desired_op == current_op then
          return desired_mode, desired_op, transitions.nothing
        else
          return desired_mode, desired_op, transitions.close_and_reopen
        end
      else
        return desired_mode, desired_op, transitions.close_and_reopen
      end
    else
      assert(false)
    end
  end
end

local function open_region_tree_node(cx, path, index, desired_mode, desired_op, field_path)
  assert(index >= 1)
  cx:state(field_path):ensure(path[index])
  local next_mode, next_op, transition = select_transition(
    cx, path, index, desired_mode, desired_op, field_path)
  local next_nid, fresh = transition(cx, path, index, field_path)
  cx:state(field_path):set_mode(path[index], next_mode)
  cx:state(field_path):set_op(path[index], next_op)
  if index >= 2 then
    return open_region_tree_node(cx, path, index-1, desired_mode, desired_op, field_path)
  end
  return next_nid, fresh
end

local function open_region_tree_top_initialize(cx, path, privilege, field_path)
  local desired_mode, desired_op = privilege_mode(privilege)
  local current_nid, fresh = open_region_tree_node(
    cx, path, #path, desired_mode, desired_op, field_path)
  if fresh then sequence_depend(cx, current_nid) end
  return current_nid
end

local function open_region_tree_top_finalize(cx, path, privilege, field_path, current_nid)
  local desired_mode, desired_op = privilege_mode(privilege)
  assert(flow.is_valid_node(current_nid))
  local next_nid
  if desired_mode == modes.write then
    next_nid = add_node(cx, cx.graph:node_label(current_nid))
    cx:state(field_path):ensure(path[1])
    cx:state(field_path):set_current(path[1], next_nid)
    cx:state(field_path):set_open(path[1], flow.null())
    cx:state(field_path):set_dirty(path[1], true)
  elseif desired_mode == modes.reduce then
    next_nid = current_nid
    current_nid = false
  end
  return current_nid, next_nid
end

local function open_region_tree_top(cx, path, privilege, field_path, initialize, finalize)
  local desired_mode, desired_op = privilege_mode(privilege)
  if not desired_mode then
    if initialize then
      -- Special case for "none" privilege: just create the node and
      -- exit without linking it up to anything.
      local next_nid = cx.graph:add_node(get_region_label(cx, path[1], field_path))
      sequence_depend(cx, next_nid)
      return data.newtuple(privilege, next_nid)
    else
      return
    end
  end

  local current_nid, next_nid
  if initialize then
    current_nid = open_region_tree_top_initialize(cx, path, privilege, field_path)
  else
    current_nid = cx:state(field_path):current(path[1])
  end
  if finalize then
    current_nid, next_nid = open_region_tree_top_finalize(
      cx, path, privilege, field_path, current_nid)
  end

  assert(current_nid or next_nid)
  assert(not current_nid or flow.is_valid_node(current_nid))
  assert(not next_nid or flow.is_valid_node(next_nid))
  return data.newtuple(privilege, current_nid, next_nid)
end

local function open_region_tree(cx, expr_type, symbol, privilege_map, initialize, finalize)
  if not initialize and not finalize then
    initialize, finalize = true, true
  end

  local region_type = cx.tree:ensure_variable(expr_type, symbol)
  assert(flow_region_tree.is_region(region_type))
  assert(is_field_map(privilege_map))
  -- FIXME: If I have to run this here, I'm probably missing one elsewhere.
  privilege_map = expand_fields(region_type, privilege_map)

  local path = data.newtuple(unpack(cx.tree:ancestors(region_type)))
  local result = new_field_map()
  for field_path, privilege in privilege_map:items() do
    local field_result = open_region_tree_top(
      cx, path, privilege, field_path, initialize, finalize)
    if field_result then
      result:insert(field_path, field_result)
    end
  end
  return result
end

local function preopen_region_tree_top(cx, path, privilege, field_path)
  local desired_mode, desired_op = privilege_mode(privilege)
  if not desired_mode then
    return
  end
  for index = #path, 2, -1 do
    -- This opens in write mode (rather than the requested mode)
    -- because preopen is occaisionally used in contexts where two
    -- disjoint regions with a common ancestor are used with
    -- conflicting privileges.
    cx:state(field_path):ensure(path[index])
    cx:state(field_path):set_mode(path[index], modes.write)
    cx:state(field_path):set_op(path[index], false)
  end
end

local function preopen_region_tree(cx, region_type, privilege_map)
  assert(flow_region_tree.is_region(region_type))
  assert(is_field_map(privilege_map))
  -- FIXME: If I have to run this here, I'm probably missing one elsewhere.
  privilege_map = expand_fields(region_type, privilege_map)

  local path = data.newtuple(unpack(cx.tree:ancestors(region_type)))
  for field_path, privilege in privilege_map:items() do
    preopen_region_tree_top(cx, path, privilege, field_path)
  end
end

-- Summarization of Privileges

local function uses_region(cx, region_type, privilege)
  local usage = data.newmap()
  usage[region_type] = privilege
  return usage
end

local function uses(cx, region_type, privilege_map)
  return expand_fields(region_type, privilege_map):map(
    function(field_path, privilege)
      return uses_region(cx, region_type, privilege)
    end)
end

local function privilege_meet_region(...)
  local usage = data.newmap()
  for _, a in pairs({...}) do
    if a then
      for region_type, privilege in a:items() do
        usage[region_type] = std.meet_privilege(usage[region_type], privilege)
      end
    end
  end
  return usage
end

local function privilege_meet(...)
  local usage = new_field_map()
  for _, a in pairs({...}) do
    assert(is_field_map(a))
    for field_path, privileges in a:items() do
      usage:insert(
        field_path,
        privilege_meet_region(usage:contains(field_path), privileges))
    end
  end
  return usage
end

local function strip_indexing(cx, region_type)
  local path = data.newtuple(unpack(cx.tree:ancestors(region_type)))
  local last_index = 0
  for index = 1, #path do
    if cx.tree:is_point(path[index]) or
      (cx.tree:has_region_index(path[index]) and
         not cx.tree:region_index(path[index]):is(ast.typed.expr.Constant))
    then
      last_index = index
    end
  end
  assert(last_index < #path)
  return path[last_index + 1]
end

local function strip_undefined(cx, region_type)
  local path = data.newtuple(unpack(cx.tree:ancestors(region_type)))
  local last_index = 0
  for index = 1, #path do
    if not cx:has_region_symbol(path[index]) then
      last_index = index
    end
  end
  assert(last_index <= #path)
  if last_index == #path then return end
  return path[last_index + 1]
end

local function privilege_summary_region(cx, usage, strip, skip_regions)
  local summary = data.newmap()
  if not usage then return summary end
  for region_type, privilege in usage:items() do
    if privilege ~= "none" or not skip_regions or not rawget(skip_regions, region_type) then
      -- FIXME: This is broken and could probably be removed.
      -- if strip then
      --   region_type = strip_indexing(cx, region_type)
      -- end
      region_type = strip_undefined(cx, region_type)

      if region_type then
        local recorded = false
        local next_summary = data.newmap()
        for other, other_privilege in summary:items() do
          local ancestor = cx.tree:lowest_common_ancestor(region_type, other)
          if ancestor and
            not (privilege == "none" or
                 other_privilege == "none" or
                 (privilege == "reads" and other_privilege == "reads" and
                    cx.tree:is_sibling(region_type, other)) or
                 (privilege == other_privilege and std.is_reduction_op(privilege) and
                    cx.tree:is_sibling(region_type, other)) or
                   not cx.tree:can_alias(region_type, other))
          then
            next_summary[ancestor] = std.meet_privilege(
              privilege,
              std.meet_privilege(
                other_privilege,
                next_summary:has(ancestor)))
            recorded = true
          else
            next_summary[other] = std.meet_privilege(other_privilege, next_summary:has(other))
          end
        end
        if not recorded then
          next_summary[region_type] = std.meet_privilege(privilege, next_summary:has(region_type))
        end
        summary = next_summary
      end
    end
  end
  return summary
end

local function privilege_summary(cx, usage, strip)
  local initial = new_field_map()
  if not usage then return initial end
  for field_path, privileges in usage:items() do
    initial:insert(field_path, privilege_summary_region(cx, privileges, strip))
  end

  -- Hack: This process can produce more "none" privileges than we'd
  -- like (because none on <> is redundant with reads <a> even though
  -- those two look like different privileges). Get a list of regions
  -- with privileges and strip any redundant regions.

  local privilege_regions = {}
  for field_path, privileges in initial:items() do
    for region_type, privilege in privileges:items() do
      if privilege ~= "none" then
        privilege_regions[region_type] = region_type
      end
    end
  end

  local summary = new_field_map()
  for field_path, privileges in usage:items() do
    summary:insert(field_path, privilege_summary_region(cx, privileges, strip, privilege_regions))
  end
  return summary
end

local function index_privileges_by_region(usage)
  -- field -> region_privileges => region -> privilege_map
  local result = data.newmap()
  assert(is_field_map(usage))
  for field_path, region_privileges in usage:items() do
    for region_type, privilege in region_privileges:items() do
      if not result:has(region_type) then
        result[region_type] = new_field_map()
      end
      result[region_type]:insert(field_path, privilege)
    end
  end
  return result
end

-- Privilege Maps

local function get_trivial_field_map(value)
  local result = new_field_map()
  result:insert(data.newtuple(), value)
  return result
end

local none = get_trivial_field_map("none")
local reads = get_trivial_field_map("reads")
local reads_writes = get_trivial_field_map("reads_writes")

local function name(value_type)
  if flow_region_tree.is_region(std.as_read(value_type)) then
    return none
  end
  return reads
end

local function get_privilege_field_map(task, region_type)
  local privileges, privilege_field_paths =
    std.find_task_privileges(region_type, task)
  local result = new_field_map()
  for i, privilege in ipairs(privileges) do
    local field_paths = privilege_field_paths[i]
    for _, field_path in ipairs(field_paths) do
      result:insert(field_path, privilege)
    end
  end

  if result:is_empty() then
    return none
  end

  return result
end

local function attach_result(privilege_map, nid)
  assert(is_field_map(privilege_map))
  local result = new_field_map()
  for k, privilege in privilege_map:items() do
    if privilege == "none" or privilege == "reads" then
      result:insert(k, data.newtuple(privilege, nid))
    elseif std.is_reduction_op(privilege) then
      result:insert(k, data.newtuple(privilege, false, nid))
    else
      assert(false)
    end
  end
  return result
end

-- Privilege Analysis and Summarization

local analyze_privileges = {}

function analyze_privileges.expr_region_root(cx, node, privilege_map)
  local region_fields = std.flatten_struct_fields(
    std.as_read(node.region.expr_type):fspace())
  local privilege_fields = terralib.newlist()
  for _, region_field in ipairs(region_fields) do
    for _, use_field in ipairs(node.fields) do
      if region_field:starts_with(use_field) then
        privilege_fields:insert(region_field)
        break
      end
    end
  end
  local field_privilege_map = new_field_map()
  for _, field_path in ipairs(privilege_fields) do
    field_privilege_map:insertall(privilege_map:prepend(field_path))
  end
  return analyze_privileges.expr(cx, node.region, field_privilege_map)
end

function analyze_privileges.expr_condition(cx, node, privilege_map)
  return analyze_privileges.expr(cx, node.value, reads)
end

function analyze_privileges.expr_id(cx, node, privilege_map)
  local expr_type = std.as_read(node.expr_type)
  if flow_region_tree.is_region(expr_type) then
    return uses(cx, expr_type, privilege_map)
  else
    if not cx.local_vars[node.value] then
      local region_type = cx.tree:intern_variable(
        node.expr_type, node.value, node.annotations, node.span)
      return uses(cx, region_type, privilege_map)
    end
  end
end

function analyze_privileges.expr_field_access(cx, node, privilege_map)
  local value_type = std.as_read(node.value.expr_type)
  local expr_type = std.as_read(node.expr_type)

  local usage
  if flow_region_tree.is_region(expr_type) then
    usage = uses(cx, expr_type, privilege_map)

    -- Make sure a symbol is available.
    if not cx:has_region_symbol(expr_type) then
      local symbol = std.newsymbol(expr_type)
      cx:intern_region(node, symbol, node.expr_type)
    end
  end

  if node.field_name == "ispace" or
    node.field_name == "bounds" or
    node.field_name == "colors"
  then
    return privilege_meet(analyze_privileges.expr(cx, node.value, none), usage)
  else
    if std.is_ref(node.expr_type) and
       std.extract_privileged_prefix(node.expr_type.refers_to_type,
                                     node.expr_type.field_path):contains(node.field_name)
    then
      privilege_map = privilege_map:prepend(node.field_name)
    end
    return privilege_meet(analyze_privileges.expr(cx, node.value, privilege_map), usage)
  end
end

function analyze_privileges.expr_index_access(cx, node, privilege_map)
  local expr_type = std.as_read(node.expr_type)
  local usage
  if flow_region_tree.is_region(expr_type) then
    usage = uses(cx, expr_type, privilege_map)

    -- Make sure a symbol is available.
    if not cx:has_region_symbol(expr_type) then
      local symbol = std.newsymbol(expr_type)
      cx:intern_region(node, symbol, node.expr_type)
    end
  elseif std.is_ref(node.expr_type) then
    local bounds = node.expr_type:bounds()
    for _, parent in ipairs(bounds) do
      local index
      -- FIXME: This causes issues with some tests.
      -- if node.value:is(ast.typed.expr.ID) and
      --   not std.is_rawref(node.value.expr_type)
      -- then
      --   index = node.value
      -- end

      -- FIXME: See flow_from_ast.expr_deref
      -- local subregion = cx.tree:intern_region_point_expr(
      --   parent, index, node.annotations, node.span)
      usage = privilege_meet(usage, uses(cx, parent, privilege_map))
    end
  end
  return privilege_meet(
    analyze_privileges.expr(cx, node.value, name(node.value.expr_type)),
    analyze_privileges.expr(cx, node.index, reads),
    usage)
end

function analyze_privileges.expr_method_call(cx, node, privilege_map)
  local usage = analyze_privileges.expr(cx, node.value, reads)
  for _, arg in ipairs(node.args) do
    usage = privilege_meet(usage, analyze_privileges.expr(cx, arg, reads))
  end
  return usage
end

function analyze_privileges.expr_call(cx, node, privilege_map)
  local usage = analyze_privileges.expr(cx, node.fn, reads)
  for i, arg in ipairs(node.args) do
    local param_type = node.fn.expr_type.parameters[i]
    local param_privilege_map
    if std.is_task(node.fn.value) and std.type_supports_privileges(param_type) then
      param_privilege_map = get_privilege_field_map(node.fn.value, param_type)
    else
      param_privilege_map = name(param_type)
    end

    usage = privilege_meet(
      usage, analyze_privileges.expr(cx, arg, param_privilege_map))
  end
  usage = data.reduce(
      privilege_meet,
      node.conditions:map(
        function(condition)
          return analyze_privileges.expr_condition(cx, condition, reads) or
            new_field_map()
        end),
      usage)
  return usage
end

function analyze_privileges.expr_cast(cx, node, privilege_map)
  return privilege_meet(analyze_privileges.expr(cx, node.fn, reads),
                        analyze_privileges.expr(cx, node.arg, name(node.arg.expr_type)))
end

function analyze_privileges.expr_ctor(cx, node, privilege_map)
  local usage = nil
  for _, field in ipairs(node.fields) do
    usage = privilege_meet(
      usage, analyze_privileges.expr(cx, field.value, name(field.value.expr_type)))
  end
  return usage
end

function analyze_privileges.expr_raw_physical(cx, node, privilege_map)
  -- assert(false) -- This case needs special handling. -- Elliott: Why?
  return privilege_meet(
    analyze_privileges.expr_region_root(cx, node.region, reads_writes))
end

function analyze_privileges.expr_raw_fields(cx, node, privilege_map)
  return analyze_privileges.expr_region_root(cx, node.region, none)
end

function analyze_privileges.expr_raw_future(cx, node, privilege_map)
  return analyze_privileges.expr(cx, node.value, none)
end

function analyze_privileges.expr_raw_value(cx, node, privilege_map)
  return analyze_privileges.expr(cx, node.value, none)
end

function analyze_privileges.expr_isnull(cx, node, privilege_map)
  return analyze_privileges.expr(cx, node.pointer, reads)
end

function analyze_privileges.expr_dynamic_cast(cx, node, privilege_map)
  return analyze_privileges.expr(cx, node.value, reads)
end

function analyze_privileges.expr_static_cast(cx, node, privilege_map)
  return analyze_privileges.expr(cx, node.value, reads)
end

function analyze_privileges.expr_unsafe_cast(cx, node, privilege_map)
  return analyze_privileges.expr(cx, node.value, reads)
end

function analyze_privileges.expr_ispace(cx, node, privilege_map)
  return privilege_meet(
    analyze_privileges.expr(cx, node.extent, reads),
    (node.start and analyze_privileges.expr(cx, node.start, reads)) or new_field_map())
end

function analyze_privileges.expr_region(cx, node, privilege_map)
  return analyze_privileges.expr(cx, node.ispace, reads)
end

function analyze_privileges.expr_partition(cx, node, privilege_map)
  return privilege_meet(
    analyze_privileges.expr(cx, node.region, none),
    analyze_privileges.expr(cx, node.coloring, reads),
    (node.colors and analyze_privileges.expr(cx, node.colors, reads)) or new_field_map())
end

function analyze_privileges.expr_partition_equal(cx, node, privilege_map)
  return privilege_meet(
    analyze_privileges.expr(cx, node.region, none),
    analyze_privileges.expr(cx, node.colors, reads))
end

function analyze_privileges.expr_partition_by_field(cx, node, privilege_map)
  return privilege_meet(
    analyze_privileges.expr_region_root(cx, node.region, reads),
    analyze_privileges.expr(cx, node.colors, reads))
end

function analyze_privileges.expr_partition_by_restriction(cx, node, privilege_map)
  return privilege_meet(
    analyze_privileges.expr(cx, node.region, reads),
    analyze_privileges.expr(cx, node.transform, reads),
    analyze_privileges.expr(cx, node.extent, reads),
    analyze_privileges.expr(cx, node.colors, reads))
end

function analyze_privileges.expr_image(cx, node, privilege_map)
  return privilege_meet(
    analyze_privileges.expr(cx, node.parent, none),
    analyze_privileges.expr(cx, node.partition, none),
    analyze_privileges.expr_region_root(cx, node.region, reads))
end

function analyze_privileges.expr_preimage(cx, node, privilege_map)
  return privilege_meet(
    analyze_privileges.expr(cx, node.parent, none),
    analyze_privileges.expr(cx, node.partition, none),
    analyze_privileges.expr_region_root(cx, node.region, reads))
end

function analyze_privileges.expr_cross_product(cx, node, privilege_map)
  return data.reduce(
    privilege_meet,
    node.args:map(
      function(arg)
        return analyze_privileges.expr(cx, arg, name(arg.expr_type)) or new_field_map()
  end))
end

function analyze_privileges.expr_cross_product_array(cx, node)
  return privilege_meet(
    analyze_privileges.expr(cx, node.lhs, none),
    analyze_privileges.expr(cx, node.colorings, reads))
end

function analyze_privileges.expr_list_slice_partition(cx, node)
  return privilege_meet(
    analyze_privileges.expr(cx, node.partition, none),
    analyze_privileges.expr(cx, node.indices, reads))
end

function analyze_privileges.expr_list_duplicate_partition(cx, node)
  return privilege_meet(
    analyze_privileges.expr(cx, node.partition, none),
    analyze_privileges.expr(cx, node.indices, reads))
end

function analyze_privileges.expr_list_cross_product(cx, node)
  return privilege_meet(
    analyze_privileges.expr(cx, node.lhs, none),
    analyze_privileges.expr(cx, node.rhs, none))
end

function analyze_privileges.expr_list_cross_product_complete(cx, node)
  return privilege_meet(
    analyze_privileges.expr(cx, node.lhs, none),
    analyze_privileges.expr(cx, node.product, none))
end

function analyze_privileges.expr_list_phase_barriers(cx, node)
  return analyze_privileges.expr(cx, node.product, none)
end

function analyze_privileges.expr_list_invert(cx, node)
  return privilege_meet(
    analyze_privileges.expr(cx, node.lhs, none),
    analyze_privileges.expr(cx, node.product, none),
    analyze_privileges.expr(cx, node.barriers, reads))
end

function analyze_privileges.expr_list_range(cx, node)
  return privilege_meet(
    analyze_privileges.expr(cx, node.start, reads),
    analyze_privileges.expr(cx, node.stop, reads))
end

function analyze_privileges.expr_list_ispace(cx, node)
  return privilege_meet(analyze_privileges.expr(cx, node.ispace, reads))
end

function analyze_privileges.expr_list_from_element(cx, node)
  return privilege_meet(
    analyze_privileges.expr(cx, node.list, reads),
    analyze_privileges.expr(cx, node.value, reads))
end

function analyze_privileges.expr_phase_barrier(cx, node)
  return analyze_privileges.expr(cx, node.value, reads)
end

function analyze_privileges.expr_dynamic_collective(cx, node)
  return analyze_privileges.expr(cx, node.arrivals, reads)
end

function analyze_privileges.expr_dynamic_collective_get_result(cx, node)
  return analyze_privileges.expr(cx, node.value, reads)
end

function analyze_privileges.expr_advance(cx, node, privilege_map)
  return analyze_privileges.expr(cx, node.value, reads)
end

function analyze_privileges.expr_adjust(cx, node, privilege_map)
  return privilege_meet(
    analyze_privileges.expr(cx, node.barrier, reads),
    analyze_privileges.expr(cx, node.value, reads))
end

function analyze_privileges.expr_arrive(cx, node, privilege_map)
  return privilege_meet(
    analyze_privileges.expr(cx, node.barrier, reads),
    (node.value and analyze_privileges.expr(cx, node.value, reads)) or new_field_map())
end

function analyze_privileges.expr_await(cx, node, privilege_map)
  return analyze_privileges.expr(cx, node.barrier, reads)
end

function analyze_privileges.expr_copy(cx, node, privilege_map)
  local dst_mode = reads_writes
  if node.op then
    dst_mode = get_trivial_field_map(reduces(node.op))
  end
  return privilege_meet(
    analyze_privileges.expr_region_root(cx, node.src, reads),
    analyze_privileges.expr_region_root(cx, node.dst, dst_mode),
    data.reduce(
      privilege_meet,
      node.conditions:map(
        function(condition)
          return analyze_privileges.expr_condition(cx, condition, reads) or
            new_field_map()
        end)))
end

function analyze_privileges.expr_fill(cx, node, privilege_map)
  return privilege_meet(
    analyze_privileges.expr_region_root(cx, node.dst, reads_writes),
    analyze_privileges.expr(cx, node.value, reads),
    data.reduce(
      privilege_meet,
      node.conditions:map(
        function(condition)
          return analyze_privileges.expr_condition(cx, condition, reads) or
            new_field_map()
        end)))
end

function analyze_privileges.expr_acquire(cx, node, privilege_map)
  return privilege_meet(
    analyze_privileges.expr_region_root(cx, node.region, reads_writes),
    data.reduce(
      privilege_meet,
      node.conditions:map(
        function(condition)
          return analyze_privileges.expr_condition(cx, condition, reads) or
            new_field_map()
        end)))
end

function analyze_privileges.expr_release(cx, node, privilege_map)
  return privilege_meet(
    analyze_privileges.expr_region_root(cx, node.region, reads_writes),
    data.reduce(
      privilege_meet,
      node.conditions:map(
        function(condition)
          return analyze_privileges.expr_condition(cx, condition, reads) or
            new_field_map()
        end)))
end

function analyze_privileges.expr_attach_hdf5(cx, node, privilege_map)
  return privilege_meet(
    analyze_privileges.expr_region_root(cx, node.region, reads_writes),
    analyze_privileges.expr(cx, node.filename, reads),
    analyze_privileges.expr(cx, node.mode, reads))
end

function analyze_privileges.expr_detach_hdf5(cx, node, privilege_map)
  return analyze_privileges.expr_region_root(cx, node.region, reads_writes)
end

function analyze_privileges.expr_unary(cx, node, privilege_map)
  return analyze_privileges.expr(cx, node.rhs, reads)
end

function analyze_privileges.expr_binary(cx, node, privilege_map)
  return privilege_meet(analyze_privileges.expr(cx, node.lhs, reads),
                        analyze_privileges.expr(cx, node.rhs, reads))
end

function analyze_privileges.expr_deref(cx, node, privilege_map)
  local usage
  if std.is_ref(node.expr_type) then
    local bounds = node.expr_type:bounds()
    for _, parent in ipairs(bounds) do
      local index
      -- FIXME: This causes issues with some tests.
      -- if node.value:is(ast.typed.expr.ID) and
      --   not std.is_rawref(node.value.expr_type)
      -- then
      --   index = node.value
      -- end

      -- FIXME: This probably *shouldn't* need to do this, but
      -- something else breaks when this is disabled.
      -- See flow_from_ast.expr_deref
      local subregion = cx.tree:intern_region_point_expr(
        parent, index, node.annotations, node.span)
      usage = privilege_meet(usage, uses(cx, subregion, privilege_map))
    end
  end
  return privilege_meet(
    analyze_privileges.expr(cx, node.value, reads),
    usage)
end

function analyze_privileges.expr_address_of(cx, node, privilege_map)
  return analyze_privileges.expr(cx, node.value, none)
end

function analyze_privileges.expr_import_ispace(cx, node, privilege_map)
  return analyze_privileges.expr(cx, node.value, reads)
end

function analyze_privileges.expr_import_region(cx, node, privilege_map)
  return privilege_meet(analyze_privileges.expr(cx, node.ispace,    reads),
                        analyze_privileges.expr(cx, node.value,     reads),
                        analyze_privileges.expr(cx, node.field_ids, reads))
end

function analyze_privileges.expr_import_partition(cx, node, privilege_map)
  return privilege_meet(analyze_privileges.expr(cx, node.region, reads),
                        analyze_privileges.expr(cx, node.colors, reads),
                        analyze_privileges.expr(cx, node.value,  reads))
end

function analyze_privileges.expr(cx, node, privilege_map)
  if node:is(ast.typed.expr.ID) then
    return analyze_privileges.expr_id(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Constant) then
    return new_field_map()

  elseif node:is(ast.typed.expr.Global) then
    return new_field_map()

  elseif node:is(ast.typed.expr.Function) then
    return new_field_map()

  elseif node:is(ast.typed.expr.FieldAccess) then
    return analyze_privileges.expr_field_access(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.IndexAccess) then
    return analyze_privileges.expr_index_access(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.MethodCall) then
    return analyze_privileges.expr_method_call(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Call) then
    return analyze_privileges.expr_call(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Cast) then
    return analyze_privileges.expr_cast(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Ctor) then
    return analyze_privileges.expr_ctor(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.RawContext) then
    return new_field_map()

  elseif node:is(ast.typed.expr.RawFields) then
    return analyze_privileges.expr_raw_fields(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.RawFuture) then
    return analyze_privileges.expr_raw_future(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.RawPhysical) then
    return analyze_privileges.expr_raw_physical(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.RawRuntime) then
    return new_field_map()

  elseif node:is(ast.typed.expr.RawTask) then
    return new_field_map()

  elseif node:is(ast.typed.expr.RawValue) then
    return analyze_privileges.expr_raw_value(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Isnull) then
    return analyze_privileges.expr_isnull(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Null) then
    return new_field_map()

  elseif node:is(ast.typed.expr.DynamicCast) then
    return analyze_privileges.expr_dynamic_cast(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.StaticCast) then
    return analyze_privileges.expr_static_cast(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.UnsafeCast) then
    return analyze_privileges.expr_unsafe_cast(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Ispace) then
    return analyze_privileges.expr_ispace(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Region) then
    return analyze_privileges.expr_region(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Partition) then
    return analyze_privileges.expr_partition(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.PartitionEqual) then
    return analyze_privileges.expr_partition_equal(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.PartitionByField) then
    return analyze_privileges.expr_partition_by_field(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.PartitionByRestriction) then
    return analyze_privileges.expr_partition_by_restriction(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Image) then
    return analyze_privileges.expr_image(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Preimage) then
    return analyze_privileges.expr_preimage(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.CrossProduct) then
    return analyze_privileges.expr_cross_product(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.CrossProductArray) then
    return analyze_privileges.expr_cross_product_array(cx, node)

  elseif node:is(ast.typed.expr.ListSlicePartition) then
    return analyze_privileges.expr_list_slice_partition(cx, node)

  elseif node:is(ast.typed.expr.ListDuplicatePartition) then
    return analyze_privileges.expr_list_duplicate_partition(cx, node)

  elseif node:is(ast.typed.expr.ListCrossProduct) then
    return analyze_privileges.expr_list_cross_product(cx, node)

  elseif node:is(ast.typed.expr.ListCrossProductComplete) then
    return analyze_privileges.expr_list_cross_product_complete(cx, node)

  elseif node:is(ast.typed.expr.ListPhaseBarriers) then
    return analyze_privileges.expr_list_phase_barriers(cx, node)

  elseif node:is(ast.typed.expr.ListInvert) then
    return analyze_privileges.expr_list_invert(cx, node)

  elseif node:is(ast.typed.expr.ListRange) then
    return analyze_privileges.expr_list_range(cx, node)

  elseif node:is(ast.typed.expr.ListIspace) then
    return analyze_privileges.expr_list_ispace(cx, node)

  elseif node:is(ast.typed.expr.ListFromElement) then
    return analyze_privileges.expr_list_from_element(cx, node)

  elseif node:is(ast.typed.expr.PhaseBarrier) then
    return analyze_privileges.expr_phase_barrier(cx, node)

  elseif node:is(ast.typed.expr.DynamicCollective) then
    return analyze_privileges.expr_dynamic_collective(cx, node)

  elseif node:is(ast.typed.expr.DynamicCollectiveGetResult) then
    return analyze_privileges.expr_dynamic_collective_get_result(cx, node)

  elseif node:is(ast.typed.expr.Advance) then
    return analyze_privileges.expr_advance(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Adjust) then
    return analyze_privileges.expr_adjust(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Arrive) then
    return analyze_privileges.expr_arrive(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Await) then
    return analyze_privileges.expr_await(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Copy) then
    return analyze_privileges.expr_copy(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Fill) then
    return analyze_privileges.expr_fill(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Acquire) then
    return analyze_privileges.expr_acquire(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Release) then
    return analyze_privileges.expr_release(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.AttachHDF5) then
    return analyze_privileges.expr_attach_hdf5(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.DetachHDF5) then
    return analyze_privileges.expr_detach_hdf5(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Unary) then
    return analyze_privileges.expr_unary(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Binary) then
    return analyze_privileges.expr_binary(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Deref) then
    return analyze_privileges.expr_deref(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.AddressOf) then
    return analyze_privileges.expr_address_of(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.ImportIspace) then
    return analyze_privileges.expr_import_ispace(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.ImportRegion) then
    return analyze_privileges.expr_import_region(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.ImportPartition) then
    return analyze_privileges.expr_import_partition(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Projection) then
    assert(false,
      "region projection is not supported by RDIR. please re-run your code with -fflow 0")

  else
    assert(false, "unexpected node type " .. tostring(node.node_type))
  end
end

function analyze_privileges.block(cx, node)
  return data.reduce(
    privilege_meet,
    node.stats:map(
      function(stat)
        return analyze_privileges.stat(cx, stat) or new_field_map()
      end))
end

function analyze_privileges.stat_if(cx, node)
  local then_cx = cx:new_local_scope()
  local then_privileges = analyze_privileges.block(then_cx, node.then_block)
  local outer_then_privileges = privilege_summary(cx, then_privileges, false)

  local else_cx = cx:new_local_scope()
  local else_privileges = analyze_privileges.block(else_cx, node.else_block)
  local outer_else_privileges = privilege_summary(cx, else_privileges, false)

  return
    privilege_meet(
      analyze_privileges.expr(cx, node.cond, reads),
      outer_then_privileges,
      data.reduce(
        privilege_meet,
        node.elseif_blocks:map(
          function(block)
            return analyze_privileges.stat_elseif(cx, block) or new_field_map()
          end)),
      outer_else_privileges)
end

function analyze_privileges.stat_elseif(cx, node)
  local block_cx = cx:new_local_scope()
  local block_privileges = analyze_privileges.block(block_cx, node.block)
  local outer_privileges = privilege_summary(cx, block_privileges, false)
  return privilege_meet(
    analyze_privileges.expr(cx, node.cond, reads),
    outer_privileges)
end

function analyze_privileges.stat_while(cx, node)
  local block_cx = cx:new_local_scope()
  local block_privileges = analyze_privileges.block(block_cx, node.block)
  local outer_privileges = privilege_summary(cx, block_privileges, true)
  return privilege_meet(
    analyze_privileges.expr(cx, node.cond, reads),
    outer_privileges)
end

function analyze_privileges.stat_for_num(cx, node)
  local block_cx = cx:new_local_scope(node.symbol)
  local block_privileges = analyze_privileges.block(block_cx, node.block)
  local outer_privileges = privilege_summary(cx, block_privileges, true)
  return
    data.reduce(
      privilege_meet,
      node.values:map(
        function(value)
          return analyze_privileges.expr(cx, value, reads) or new_field_map()
        end),
      outer_privileges)
end

function analyze_privileges.stat_for_list(cx, node)
  local block_cx = cx:new_local_scope(node.symbol)
  local block_privileges = analyze_privileges.block(block_cx, node.block)
  local outer_privileges = privilege_summary(cx, block_privileges, true)
  return privilege_meet(
    analyze_privileges.expr(cx, node.value, name(node.value.expr_type)),
    outer_privileges)
end

function analyze_privileges.stat_repeat(cx, node)
  local block_cx = cx:new_local_scope()
  local block_privileges = analyze_privileges.block(block_cx, node.block)
  local outer_privileges = privilege_summary(cx, block_privileges, true)
  return privilege_meet(
    outer_privileges,
    analyze_privileges.expr(cx, node.until_cond, reads))
end

function analyze_privileges.stat_must_epoch(cx, node)
  local block_cx = cx:new_local_scope()
  local block_privileges = analyze_privileges.block(block_cx, node.block)
  local outer_privileges = privilege_summary(cx, block_privileges, false)
  return outer_privileges
end

function analyze_privileges.stat_block(cx, node)
  local block_cx = cx:new_local_scope()
  local block_privileges = analyze_privileges.block(block_cx, node.block)
  local outer_privileges = privilege_summary(cx, block_privileges, false)
  return outer_privileges
end

function analyze_privileges.stat_var(cx, node)
  local var_type = std.rawref(&node.type)
  if not flow_region_tree.is_region(std.as_read(var_type)) or
    not cx:has_region_symbol(std.as_read(var_type))
  then
    cx:intern_region(node, node.symbol, var_type)
  end

  return node.value and analyze_privileges.expr(cx, node.value, name(node.value.expr_type)) or
         new_field_map()
end

function analyze_privileges.stat_var_unpack(cx, node)
  for i, var_symbol in ipairs(node.symbols) do
    local var_type = std.rawref(&node.field_types[i])
    cx:intern_region(node, var_symbol, var_type)
  end

  return analyze_privileges.expr(cx, node.value, name(node.value.expr_type))
end

function analyze_privileges.stat_return(cx, node) 
  if node.value then
    return analyze_privileges.expr(cx, node.value, name(node.value.expr_type))
  else
    return nil
  end
end

function analyze_privileges.stat_assignment(cx, node)
  return
    privilege_meet(
      analyze_privileges.expr(cx, node.lhs, reads_writes) or new_field_map(),
      analyze_privileges.expr(cx, node.rhs, reads) or new_field_map())
end

function analyze_privileges.stat_reduce(cx, node)
  local op = get_trivial_field_map(reduces(node.op))
  return
    privilege_meet(
      analyze_privileges.expr(cx, node.lhs, op) or new_field_map(),
      analyze_privileges.expr(cx, node.rhs, reads) or new_field_map())
end

function analyze_privileges.stat_expr(cx, node)
  return analyze_privileges.expr(cx, node.expr, name(node.expr.expr_type))
end

function analyze_privileges.stat_raw_delete(cx, node)
  return analyze_privileges.expr(cx, node.value, name(node.value.expr_type))
end

function analyze_privileges.stat(cx, node)
  if node:is(ast.typed.stat.If) then
    return analyze_privileges.stat_if(cx, node)

  elseif node:is(ast.typed.stat.While) then
    return analyze_privileges.stat_while(cx, node)

  elseif node:is(ast.typed.stat.ForNum) then
    return analyze_privileges.stat_for_num(cx, node)

  elseif node:is(ast.typed.stat.ForList) then
    return analyze_privileges.stat_for_list(cx, node)

  elseif node:is(ast.typed.stat.Repeat) then
    return analyze_privileges.stat_repeat(cx, node)

  elseif node:is(ast.typed.stat.MustEpoch) then
    return analyze_privileges.stat_must_epoch(cx, node)

  elseif node:is(ast.typed.stat.Block) then
    return analyze_privileges.stat_block(cx, node)

  elseif node:is(ast.typed.stat.Var) then
    return analyze_privileges.stat_var(cx, node)

  elseif node:is(ast.typed.stat.VarUnpack) then
    return analyze_privileges.stat_var_unpack(cx, node)

  elseif node:is(ast.typed.stat.Return) then
    return analyze_privileges.stat_return(cx, node)

  elseif node:is(ast.typed.stat.Break) then
    return nil

  elseif node:is(ast.typed.stat.Assignment) then
    return analyze_privileges.stat_assignment(cx, node)

  elseif node:is(ast.typed.stat.Reduce) then
    return analyze_privileges.stat_reduce(cx, node)

  elseif node:is(ast.typed.stat.Expr) then
    return analyze_privileges.stat_expr(cx, node)

  elseif node:is(ast.typed.stat.RawDelete) then
    return analyze_privileges.stat_raw_delete(cx, node)

  elseif node:is(ast.typed.stat.Fence) then
    return nil

  elseif node:is(ast.typed.stat.ParallelPrefix) then
    return nil

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

-- AST -> Dataflow IR

local flow_from_ast = {}

local function as_stat(cx, args, label)
  local compute_nid = add_node(cx, label)
  add_args(cx, compute_nid, args)
  sequence_depend(cx, compute_nid)
  return sequence_advance(cx, compute_nid)
end

local function as_opaque_stat(cx, node)
  return as_stat(cx, terralib.newlist(), flow.node.Opaque { action = node })
end

local function as_block_stat(cx, block, args, annotations, span)
  return as_stat(cx, args, flow.node.ctrl.Block {
    block = block,
    annotations = annotations,
    span = span,
  })
end

local function as_while_body_stat(cx, block, args, annotations, span)
  return as_stat(cx, args, flow.node.ctrl.WhileBody {
    block = block,
    annotations = annotations,
    span = span,
  })
end

local function as_while_loop_stat(cx, block, args, annotations, span)
  return as_stat(cx, args, flow.node.ctrl.WhileLoop {
    block = block,
    annotations = annotations,
    span = span,
  })
end

local function as_fornum_stat(cx, symbol, block, args, annotations, span)
  return as_stat(cx, args, flow.node.ctrl.ForNum {
    symbol = symbol,
    block = block,
    annotations = annotations,
    span = span,
  })
end

local function as_forlist_stat(cx, symbol, block, args, annotations, span)
  return as_stat(cx, args, flow.node.ctrl.ForList {
    symbol = symbol,
    block = block,
    annotations = annotations,
    span = span,
  })
end

local function as_assignment_stat(cx, args, annotations, span)
  return as_stat(cx, args, flow.node.Assignment {
    annotations = annotations,
    span = span,
  })
end

local function as_reduce_stat(cx, op, args, annotations, span)
  return as_stat(cx, args, flow.node.Reduce {
    op = op,
    annotations = annotations,
    span = span,
  })
end

local function as_raw_opaque_expr(cx, node, args, privilege_map)
  local label = flow.node.Opaque { action = node }
  local compute_nid = add_node(cx, label)
  local result_nid = add_result(
    cx, compute_nid, std.as_read(node.expr_type), node.annotations, node.span)
  add_args(cx, compute_nid, args)
  sequence_depend(cx, compute_nid)
  return attach_result(privilege_map, result_nid)
end

local function as_opaque_expr(cx, generator, args, privilege_map)
  local arg_nids = args:map(function(arg) return as_nid(cx, arg) end)
  local arg_expr_nids = arg_nids:map(
    function(arg_nid)
      local arg_label = cx.graph:node_label(arg_nid)
      if arg_label:is(flow.node.data.Scalar) and arg_label.fresh then
        local arg_expr_nid = cx.graph:immediate_predecessor(arg_nid)
        if cx.graph:node_label(arg_expr_nid):is(flow.node.Opaque) then
          return arg_expr_nid
        end
      end
      return false
    end)
  local arg_asts = data.zip(arg_nids, arg_expr_nids):map(
    function(nids)
      local arg_nid, arg_expr_nid = unpack(nids)
      if arg_expr_nid then
        return cx.graph:node_label(arg_expr_nid).action
      else
        return cx.graph:node_label(arg_nid).value
      end
    end)

  local node = generator(unpack(arg_asts))
  local label = flow.node.Opaque { action = node }
  local compute_nid = add_node(cx, label)
  local result_nid = add_result(
    cx, compute_nid, std.as_read(node.expr_type), node.annotations, node.span)
  add_args(cx, compute_nid, args)
  sequence_depend(cx, compute_nid)
  local next_port = #args + 1
  for i, arg_nid in ipairs(arg_nids) do
    local arg_expr_nid = arg_expr_nids[i]
    if arg_expr_nid then
      local arg_expr_inputs = cx.graph:incoming_edges(arg_expr_nid)
      for _, edge in ipairs(arg_expr_inputs) do
        local port = edge.to_port
        if port > 0 then
          port = next_port
          next_port = next_port + 1
        end
        cx.graph:add_edge(edge.label, edge.from_node, edge.from_port,
                          compute_nid, port)
      end
      cx.graph:remove_node(arg_nid)
      cx.graph:remove_node(arg_expr_nid)
    end
  end
  return attach_result(privilege_map, result_nid)
end

local function as_call_expr(cx, args, opaque, replicable, expr_type, annotations, span, privilege_map)
  local label = flow.node.Task {
    opaque = opaque,
    replicable = replicable,
    expr_type = expr_type,
    annotations = annotations,
    span = span,
  }
  local compute_nid = add_node(cx, label)
  local result_nid = add_result(cx, compute_nid, expr_type, annotations, span)
  add_args(cx, compute_nid, args)
  sequence_depend(cx, compute_nid)
  return attach_result(privilege_map, result_nid)
end

local function as_copy_expr(cx, args, src_field_paths, dst_field_paths,
                            op, annotations, span, privilege_map)
  local label = flow.node.Copy {
    src_field_paths = src_field_paths,
    dst_field_paths = dst_field_paths,
    op = op,
    annotations = annotations,
    span = span,
  }
  local compute_nid = add_node(cx, label)
  add_args(cx, compute_nid, args)
  sequence_depend(cx, compute_nid)
  return attach_result(privilege_map, compute_nid)
end

local function as_fill_expr(cx, args, dst_field_paths,
                            annotations, span, privilege_map)
  local label = flow.node.Fill {
    dst_field_paths = dst_field_paths,
    annotations = annotations,
    span = span,
  }
  local compute_nid = add_node(cx, label)
  add_args(cx, compute_nid, args)
  sequence_depend(cx, compute_nid)
  return attach_result(privilege_map, compute_nid)
end

local function as_acquire_expr(cx, args, field_paths,
                               annotations, span, privilege_map)
  local label = flow.node.Acquire {
    field_paths = field_paths,
    annotations = annotations,
    span = span,
  }
  local compute_nid = add_node(cx, label)
  add_args(cx, compute_nid, args)
  sequence_depend(cx, compute_nid)
  return attach_result(privilege_map, compute_nid)
end

local function as_release_expr(cx, args, field_paths,
                               annotations, span, privilege_map)
  local label = flow.node.Release {
    field_paths = field_paths,
    annotations = annotations,
    span = span,
  }
  local compute_nid = add_node(cx, label)
  add_args(cx, compute_nid, args)
  sequence_depend(cx, compute_nid)
  return attach_result(privilege_map, compute_nid)
end

local function as_attach_hdf5_expr(cx, args, field_paths,
                                   annotations, span, privilege_map)
  local label = flow.node.AttachHDF5 {
    field_paths = field_paths,
    annotations = annotations,
    span = span,
  }
  local compute_nid = add_node(cx, label)
  add_args(cx, compute_nid, args)
  sequence_depend(cx, compute_nid)
  return attach_result(privilege_map, compute_nid)
end

local function as_detach_hdf5_expr(cx, args, field_paths,
                                   annotations, span, privilege_map)
  local label = flow.node.DetachHDF5 {
    field_paths = field_paths,
    annotations = annotations,
    span = span,
  }
  local compute_nid = add_node(cx, label)
  add_args(cx, compute_nid, args)
  sequence_depend(cx, compute_nid)
  return attach_result(privilege_map, compute_nid)
end

local function as_binary_expr(cx, op, args, expr_type, annotations, span,
                              privilege_map)
  local label = flow.node.Binary {
    op = op,
    expr_type = expr_type,
    annotations = annotations,
    span = span,
  }
  local compute_nid = add_node(cx, label)
  local result_nid = add_result(cx, compute_nid, expr_type, annotations, span)
  add_args(cx, compute_nid, args)
  sequence_depend(cx, compute_nid)
  return attach_result(privilege_map, result_nid)
end

local function as_cast_expr(cx, args, expr_type, annotations, span,
                            privilege_map)
  local label = flow.node.Cast {
    expr_type = expr_type,
    annotations = annotations,
    span = span,
  }
  local compute_nid = add_node(cx, label)
  local result_nid = add_result(cx, compute_nid, expr_type, annotations, span)
  add_args(cx, compute_nid, args)
  sequence_depend(cx, compute_nid)
  return attach_result(privilege_map, result_nid)
end

local function as_index_expr(cx, args, result, expr_type, annotations, span)
  local label = flow.node.IndexAccess {
    expr_type = expr_type,
    annotations = annotations,
    span = span,
  }
  local compute_nid = add_node(cx, label)
  for _, value in result:items() do
    local _, input_nid, output_nid = unpack(value)
    add_name_edge(cx, compute_nid, input_nid or output_nid)
  end
  add_args(cx, compute_nid, args)
  sequence_depend(cx, compute_nid)
  return result
end

local function as_deref_expr(cx, args, result_nid, expr_type, annotations, span,
                             privilege_map)
  local label = flow.node.Deref {
    expr_type = expr_type,
    annotations = annotations,
    span = span,
  }
  local compute_nid = add_node(cx, label)
  add_name_edge(cx, compute_nid, result_nid)
  add_args(cx, compute_nid, args)
  sequence_depend(cx, compute_nid)
  return attach_result(privilege_map, result_nid)
end

function flow_from_ast.expr_region_root(cx, node, privilege_map, init_only)
  local region_fields = std.flatten_struct_fields(
    std.as_read(node.region.expr_type):fspace())
  local privilege_fields = terralib.newlist()
  for _, region_field in ipairs(region_fields) do
    for _, use_field in ipairs(node.fields) do
      if region_field:starts_with(use_field) then
        privilege_fields:insert(region_field)
        break
      end
    end
  end
  local field_privilege_map = new_field_map()
  for _, field_path in ipairs(privilege_fields) do
    field_privilege_map:insertall(privilege_map:prepend(field_path))
  end

  return flow_from_ast.expr(cx, node.region, field_privilege_map)
end

function flow_from_ast.expr_condition(cx, node, privilege_map, init_only)
  local value = flow_from_ast.expr(cx, node.value, reads)
  return as_opaque_expr(
    cx,
    function(v1) return node { value = v1 } end,
    terralib.newlist({value}),
    privilege_map)
end

function flow_from_ast.expr_id(cx, node, privilege_map, init_only)
  -- FIXME: Why am I getting vars typed as unit?
  if std.as_read(node.expr_type) == terralib.types.unit then
    return new_field_map()
  end
  return open_region_tree(cx, node.expr_type, node.value, privilege_map, init_only)
end

function flow_from_ast.expr_constant(cx, node, privilege_map, init_only)
  return attach_result(
    privilege_map,
    cx.graph:add_node(flow.node.Constant { value = node }))
end

function flow_from_ast.expr_global(cx, node, privilege_map, init_only)
  return attach_result(
    privilege_map,
    cx.graph:add_node(flow.node.Global { value = node }))
end

function flow_from_ast.expr_function(cx, node, privilege_map, init_only)
  return attach_result(
    privilege_map,
    cx.graph:add_node(flow.node.Function { value = node }))
end

function flow_from_ast.expr_field_access(cx, node, privilege_map, init_only)
  local value_type = std.as_read(node.value.expr_type)
  local value

  local field_privilege_map = privilege_map:prepend(node.field_name)
  if node.field_name == "ispace" or
    node.field_name == "bounds" or
    node.field_name == "colors"
  then
    value = flow_from_ast.expr(cx, node.value, none)
  else
    value = flow_from_ast.expr(cx, node.value, field_privilege_map)
  end
  return as_opaque_expr(
    cx,
    function(v1) return node { value = v1 } end,
    terralib.newlist({value}),
    field_privilege_map)
end

function flow_from_ast.expr_index_access(cx, node, privilege_map, init_only)
  local expr_type = std.as_read(node.expr_type)
  local value = flow_from_ast.expr(cx, node.value, name(node.value.expr_type))
  local index = flow_from_ast.expr(cx, node.index, reads)

  if flow_region_tree.is_region(expr_type) then
    local inputs = terralib.newlist({value, index})
    local region = open_region_tree(cx, node.expr_type, nil, privilege_map, init_only)
    return as_index_expr(
      cx, inputs, region, expr_type, node.annotations, node.span)
  end

  return as_opaque_expr(
    cx,
    function(v1, v2) return node { value = v1, index = v2 } end,
    terralib.newlist({value, index}),
    privilege_map)
end

function flow_from_ast.expr_method_call(cx, node, privilege_map, init_only)
  local value = flow_from_ast.expr(cx, node.value, reads)
  local args = node.args:map(function(arg) return flow_from_ast.expr(cx, arg, reads) end)
  local inputs = terralib.newlist({value})
  inputs:insertall(args)
  return as_raw_opaque_expr(
    cx,
    node {
      value = as_ast(cx, value),
      args = args:map(function(arg) return as_ast(cx, arg) end),
    },
    inputs, privilege_map)
end

function flow_from_ast.expr_call(cx, node, privilege_map, init_only)
  local fn = flow_from_ast.expr(cx, node.fn, reads)
  local inputs = terralib.newlist({fn})

  local privileges = terralib.newlist()
  for i, arg in ipairs(node.args) do
    local param_type = node.fn.expr_type.parameters[i]
    local param_privilege_map
    if std.is_task(node.fn.value) and std.type_supports_privileges(param_type) then
      param_privilege_map = get_privilege_field_map(node.fn.value, param_type)
    else
      param_privilege_map = name(param_type)
    end
    privileges:insert(param_privilege_map)
  end

  -- Perform a two-pass analysis of region arguments. This avoid loops
  -- in the graph when two inputs are not statically known to be
  -- disjoint.
  for i, arg in ipairs(node.args) do
    local param_privilege_map = privileges[i]
    inputs[i+1] = flow_from_ast.expr(cx, arg, param_privilege_map, true)
  end
  for i, arg in ipairs(node.args) do
    local param_privilege_map = privileges[i]
    if arg:is(ast.typed.expr.ID) then
      inputs[i+1]:insertall(open_region_tree(cx, arg.expr_type, arg.value, param_privilege_map, false, true))
    elseif flow_region_tree.is_region(std.as_read(arg.expr_type)) then
      inputs[i+1]:insertall(open_region_tree(cx, arg.expr_type, nil, param_privilege_map, false, true))
    end
  end

  inputs:insertall(
    node.conditions:map(
      function(condition)
        return flow_from_ast.expr_condition(cx, condition, reads)
      end))

  return as_call_expr(
    cx, inputs,
    not std.is_task(node.fn.value),
    node.replicable,
    std.as_read(node.expr_type),
    node.annotations, node.span,
    privilege_map)
end

function flow_from_ast.expr_cast(cx, node, privilege_map, init_only)
  local fn = flow_from_ast.expr(cx, node.fn, reads)
  local arg = flow_from_ast.expr(cx, node.arg, name(node.arg.expr_type))
  local inputs = terralib.newlist({fn, arg})
  return as_cast_expr(
    cx, inputs, node.expr_type, node.annotations, node.span,
    privilege_map)
end

function flow_from_ast.expr_ctor(cx, node, privilege_map, init_only)
  local values = node.fields:map(
    function(field) return flow_from_ast.expr(cx, field.value, name(field.value.expr_type)) end)
  local fields = data.zip(node.fields, values):map(
    function(pair)
      local field, value = unpack(pair)
      return field { value = as_ast(cx, value) }
    end)
  return as_raw_opaque_expr(
    cx,
    node { fields = fields },
    values, privilege_map)
end

function flow_from_ast.expr_raw_context(cx, node, privilege_map, init_only)
  return as_opaque_expr(
    cx,
    function() return node end,
    terralib.newlist(),
    privilege_map)
end

function flow_from_ast.expr_raw_fields(cx, node, privilege_map, init_only)
  local region = flow_from_ast.expr(cx, node.region, none)
  return as_opaque_expr(
    cx,
    function(v1) return node { region = v1 } end,
    terralib.newlist({region}),
    privilege_map)
end

function flow_from_ast.expr_raw_value(cx, node, privilege_map, init_only)
  local value = flow_from_ast.expr(cx, node.value, name(node.value.expr_type))
  return as_opaque_expr(
    cx,
    function(v1) return node { value = v1 } end,
    terralib.newlist({value}),
    privilege_map)
end

function flow_from_ast.expr_raw_physical(cx, node, privilege_map, init_only)
  local region = flow_from_ast.expr(cx, node.region, reads_writes)
  return as_opaque_expr(
    cx,
    function(v1) return node { region = v1 } end,
    terralib.newlist({region}),
    privilege_map)
end

function flow_from_ast.expr_raw_runtime(cx, node, privilege_map, init_only)
  return as_opaque_expr(
    cx,
    function() return node end,
    terralib.newlist(),
    privilege_map)
end

function flow_from_ast.expr_raw_task(cx, node, privilege_map, init_only)
  return as_opaque_expr(
    cx,
    function() return node end,
    terralib.newlist(),
    privilege_map)
end

function flow_from_ast.expr_raw_value(cx, node, privilege_map, init_only)
  local value = flow_from_ast.expr(cx, node.value, name(node.value.expr_type))
  return as_opaque_expr(
    cx,
    function(v1) return node { value = v1 } end,
    terralib.newlist({value}),
    privilege_map)
end

function flow_from_ast.expr_isnull(cx, node, privilege_map, init_only)
  local pointer = flow_from_ast.expr(cx, node.pointer, reads)
  return as_opaque_expr(
    cx,
    function(v1) return node { pointer = v1 } end,
    terralib.newlist({pointer}),
    privilege_map)
end

function flow_from_ast.expr_null(cx, node, privilege_map, init_only)
  return as_opaque_expr(
    cx,
    function() return node end,
    terralib.newlist(),
    privilege_map)
end

function flow_from_ast.expr_dynamic_cast(cx, node, privilege_map, init_only)
  local value = flow_from_ast.expr(cx, node.value, reads)
  return as_opaque_expr(
    cx,
    function(v1) return node { value = v1 } end,
    terralib.newlist({value}),
    privilege_map)
end

function flow_from_ast.expr_static_cast(cx, node, privilege_map, init_only)
  local value = flow_from_ast.expr(cx, node.value, reads)
  return as_opaque_expr(
    cx,
    function(v1) return node { value = v1 } end,
    terralib.newlist({value}),
    privilege_map)
end

function flow_from_ast.expr_unsafe_cast(cx, node, privilege_map, init_only)
  local value = flow_from_ast.expr(cx, node.value, reads)
  return as_opaque_expr(
    cx,
    function(v1) return node { value = v1 } end,
    terralib.newlist({value}),
    privilege_map)
end

function flow_from_ast.expr_ispace(cx, node, privilege_map, init_only)
  local extent = flow_from_ast.expr(cx, node.extent, reads)
  local start = node.start and flow_from_ast.expr(cx, node.start, reads)
  return as_opaque_expr(
    cx,
    function(v1, v2) return node { extent = v1, start = v2 or false } end,
    terralib.newlist({extent, start or nil}),
    privilege_map)
end

function flow_from_ast.expr_region(cx, node, privilege_map, init_only)
  local ispace = flow_from_ast.expr(cx, node.ispace, reads)
  return as_opaque_expr(
    cx,
    function(v1) return node { ispace = v1 } end,
    terralib.newlist({ispace}),
    privilege_map)
end

function flow_from_ast.expr_partition(cx, node, privilege_map, init_only)
  local region = flow_from_ast.expr(cx, node.region, none)
  local coloring = flow_from_ast.expr(cx, node.coloring, reads)
  local colors = flow_from_ast.expr(cx, node.colors, reads)

  -- Make sure a symbol is available.
  local expr_type = std.as_read(node.expr_type)
  if not cx:has_region_symbol(expr_type) then
    local symbol = std.newsymbol(expr_type)
    cx:intern_region(node, symbol, node.expr_type)
  end

  return as_opaque_expr(
    cx,
    function(v1, v2, v3) return node { region = v1, coloring = v2, colors = v3 } end,
    terralib.newlist({region, coloring, colors}),
    privilege_map)
end

function flow_from_ast.expr_partition_equal(cx, node, privilege_map, init_only)
  local region = flow_from_ast.expr(cx, node.region, none)
  local colors = flow_from_ast.expr(cx, node.colors, reads)

  -- Make sure a symbol is available.
  local expr_type = std.as_read(node.expr_type)
  if not cx:has_region_symbol(expr_type) then
    local symbol = std.newsymbol(expr_type)
    cx:intern_region(node, symbol, node.expr_type)
  end

  return as_opaque_expr(
    cx,
    function(v1, v2) return node { region = v1, colors = v2 } end,
    terralib.newlist({region, colors}),
    privilege_map)
end

function flow_from_ast.expr_partition_by_field(cx, node, privilege_map, init_only)
  local region = flow_from_ast.expr_region_root(cx, node.region, none)
  local colors = flow_from_ast.expr(cx, node.colors, reads)

  -- Make sure a symbol is available.
  local expr_type = std.as_read(node.expr_type)
  if not cx:has_region_symbol(expr_type) then
    local symbol = std.newsymbol(expr_type)
    cx:intern_region(node, symbol, node.expr_type)
  end

  return as_opaque_expr(
    cx,
    function(v1, v2) return node { region = v1, colors = v2 } end,
    terralib.newlist({region, colors}),
    privilege_map)
end

function flow_from_ast.expr_partition_by_restriction(cx, node, privilege_map, init_only)
  local region = flow_from_ast.expr(cx, node.region, none)
  local transform = flow_from_ast.expr(cx, node.transform, none)
  local extent = flow_from_ast.expr(cx, node.extent, none)
  local colors = flow_from_ast.expr(cx, node.colors, reads)

  -- Make sure a symbol is available.
  local expr_type = std.as_read(node.expr_type)
  if not cx:has_region_symbol(expr_type) then
    local symbol = std.newsymbol(expr_type)
    cx:intern_region(node, symbol, node.expr_type)
  end

  return as_opaque_expr(
    cx,
    function(v1, v2, v3, v4) return node { region = v1, transform = v2, extent = v3, colors = v4 } end,
    terralib.newlist({region, transform, extent, colors}),
    privilege_map)
end

function flow_from_ast.expr_image(cx, node, privilege_map, init_only)
  local parent = flow_from_ast.expr(cx, node.parent, none)
  local partition = flow_from_ast.expr(cx, node.partition, none)
  local region = flow_from_ast.expr_region_root(cx, node.region, reads)

  -- Make sure a symbol is available.
  local expr_type = std.as_read(node.expr_type)
  if not cx:has_region_symbol(expr_type) then
    local symbol = std.newsymbol(expr_type)
    cx:intern_region(node, symbol, node.expr_type)
  end

  return as_opaque_expr(
    cx,
    function(v1, v2, v3) return node { parent = v1, partition = v2, region = v3 } end,
    terralib.newlist({parent, partition, region}),
    privilege_map)
end

function flow_from_ast.expr_preimage(cx, node, privilege_map, init_only)
  local parent = flow_from_ast.expr(cx, node.parent, none)
  local partition = flow_from_ast.expr(cx, node.partition, none)
  local region = flow_from_ast.expr_region_root(cx, node.region, reads)

  -- Make sure a symbol is available.
  local expr_type = std.as_read(node.expr_type)
  if not cx:has_region_symbol(expr_type) then
    local symbol = std.newsymbol(expr_type)
    cx:intern_region(node, symbol, node.expr_type)
  end

  return as_opaque_expr(
    cx,
    function(v1, v2, v3) return node { parent = v1, partition = v2, region = v3 } end,
    terralib.newlist({parent, partition, region}),
    privilege_map)
end

function flow_from_ast.expr_cross_product(cx, node, privilege_map, init_only)
  local args = node.args:map(function(arg) return flow_from_ast.expr(cx, arg, none) end)

  -- Make sure a symbol is available.
  local expr_type = std.as_read(node.expr_type)
  if not cx:has_region_symbol(expr_type) then
    local symbol = std.newsymbol(expr_type)
    cx:intern_region(node, symbol, node.expr_type)
  end

  return as_opaque_expr(
    cx,
    function(...) return node { args = terralib.newlist({...}) } end,
    args,
    privilege_map)
end

function flow_from_ast.expr_advance(cx, node, privilege_map, init_only)
  local value = flow_from_ast.expr(cx, node.value, reads)
  return as_opaque_expr(
    cx,
    function(v1) return node { value = v1 } end,
    terralib.newlist({value}),
    privilege_map)
end

function flow_from_ast.expr_adjust(cx, node, privilege_map, init_only)
  local barrier = flow_from_ast.expr(cx, node.barrier, reads)
  local value = flow_from_ast.expr(cx, node.value, reads)
  return as_opaque_expr(
    cx,
    function(v1, v2) return node { barrier = v1, value = v2 } end,
    terralib.newlist({barrier, value}),
    privilege_map)
end

function flow_from_ast.expr_arrive(cx, node, privilege_map, init_only)
  local barrier = flow_from_ast.expr(cx, node.barrier, reads)
  local value = node.value and flow_from_ast.expr(cx, node.value, reads)
  return as_opaque_expr(
    cx,
    function(v1, v2) return node { barrier = v1, value = v2 or false } end,
    terralib.newlist({barrier, value or nil}),
    privilege_map)
end

function flow_from_ast.expr_await(cx, node, privilege_map, init_only)
  local barrier = flow_from_ast.expr(cx, node.barrier, reads)
  return as_opaque_expr(
    cx,
    function(v1) return node { barrier = v1 } end,
    terralib.newlist({barrier}),
    privilege_map)
end

function flow_from_ast.expr_copy(cx, node, privilege_map, init_only)
  local dst_mode = reads_writes
  if node.op then
    dst_mode = get_trivial_field_map(reduces(node.op))
  end

  local src = flow_from_ast.expr_region_root(cx, node.src, reads)
  local dst = flow_from_ast.expr_region_root(cx, node.dst, dst_mode)
  local conditions = node.conditions:map(
    function(condition)
      return flow_from_ast.expr_condition(cx, condition, reads)
    end)

  local inputs = terralib.newlist({src, dst, unpack(conditions)})
  return as_copy_expr(
    cx, inputs, node.src.fields, node.dst.fields, node.op,
    node.annotations, node.span, privilege_map)
end

function flow_from_ast.expr_fill(cx, node, privilege_map, init_only)
  local dst = flow_from_ast.expr_region_root(cx, node.dst, reads_writes)
  local value = flow_from_ast.expr(cx, node.value, reads)
  local conditions = node.conditions:map(
    function(condition)
      return flow_from_ast.expr_condition(cx, condition, reads)
    end)

  local inputs = terralib.newlist({dst, value, unpack(conditions)})
  return as_fill_expr(
    cx, inputs, node.dst.fields,
    node.annotations, node.span, privilege_map)
end

function flow_from_ast.expr_acquire(cx, node, privilege_map, init_only)
  local region = flow_from_ast.expr_region_root(cx, node.region, reads_writes)
  local conditions = node.conditions:map(
    function(condition)
      return flow_from_ast.expr_condition(cx, condition, reads)
    end)

  local inputs = terralib.newlist({region, unpack(conditions)})
  return as_acquire_expr(
    cx, inputs, node.region.fields,
    node.annotations, node.span, privilege_map)
end

function flow_from_ast.expr_release(cx, node, privilege_map, init_only)
  local region = flow_from_ast.expr_region_root(cx, node.region, reads_writes)
  local conditions = node.conditions:map(
    function(condition)
      return flow_from_ast.expr_condition(cx, condition, reads)
    end)

  local inputs = terralib.newlist({region, unpack(conditions)})
  return as_release_expr(
    cx, inputs, node.region.fields,
    node.annotations, node.span, privilege_map)
end

function flow_from_ast.expr_attach_hdf5(cx, node, privilege_map, init_only)
  local region = flow_from_ast.expr_region_root(cx, node.region, reads_writes)
  local filename = flow_from_ast.expr(cx, node.filename, reads)
  local mode = flow_from_ast.expr(cx, node.mode, reads)
  local fm = node.field_map and flow_from_ast.expr(cx, node.field_map, reads)

  local inputs = terralib.newlist({region, filename, mode})
  if fm then inputs:insert(fm) end
  return as_attach_hdf5_expr(
    cx, inputs, node.region.fields,
    node.annotations, node.span, privilege_map)
end

function flow_from_ast.expr_detach_hdf5(cx, node, privilege_map, init_only)
  local region = flow_from_ast.expr_region_root(cx, node.region, reads_writes)

  local inputs = terralib.newlist({region})
  return as_detach_hdf5_expr(
    cx, inputs, node.region.fields,
    node.annotations, node.span, privilege_map)
end

function flow_from_ast.expr_unary(cx, node, privilege_map, init_only)
  local rhs = flow_from_ast.expr(cx, node.rhs, reads)
  return as_opaque_expr(
    cx,
    function(v1) return node { rhs = v1 } end,
    terralib.newlist({rhs}),
    privilege_map)
end

function flow_from_ast.expr_binary(cx, node, privilege_map, init_only)
  local lhs = flow_from_ast.expr(cx, node.lhs, reads)
  local rhs = flow_from_ast.expr(cx, node.rhs, reads)
  local inputs = terralib.newlist({lhs, rhs})
  return as_binary_expr(
    cx, node.op, inputs, node.expr_type, node.annotations, node.span,
    privilege_map)
end

function flow_from_ast.expr_deref(cx, node, privilege_map, init_only)
  local value = flow_from_ast.expr(cx, node.value, reads)
  -- FIXME: So it turns out that dereferencing, and point regions in
  -- general, are very messed up. There are a couple of moving parts
  -- that make this difficult. First, references correspond to
  -- implicit references on point regions. But the regions in question
  -- are virtual, and nothing that expects a reference can actually
  -- deal with a region. The previous approach was to hack around this
  -- by creating a fresh symbol (of the type of reference when coerced
  -- to an r-value), but that has a bunch of issues:
  --
  --  1. You can't handle l-values at all.
  --  2. Struct values break because open_region_tree blasts out all
  --     the fields and can't put them back together again.
  --  3. If the value is consumed by an opaque node, you need some way
  --     of capturing the pointer (and then separately making sure the
  --     opaque node knows to dereference it).
  --
  -- For now I'm going to generate opaque nodes for all dereferences
  -- and revisit the semantics of point regions at a later time.

  -- local value_type = std.as_read(node.value.expr_type)
  -- if std.is_bounded_type(value_type) then
  --   local bounds = value_type:bounds()
  --   if #bounds == 1 and std.is_region(bounds[1]) then
  --     local parent = bounds[1]
  --     local index
  --     -- FIXME: This causes issues with some tests.
  --     -- if node.value:is(ast.typed.expr.ID) and
  --     --   not std.is_rawref(node.value.expr_type)
  --     -- then
  --     --   index = node.value
  --     -- end
  --     local subregion, symbol = cx.tree:intern_region_point_expr(
  --       parent, index, node.annotations, node.span)
  --     if not cx:has_region_symbol(subregion) then
  --       cx:intern_region_point_expr(node, symbol, subregion)
  --     end

  --     local inputs = terralib.newlist({value})
  --     local region = open_region_tree(cx, subregion, nil, privilege_map)
  --     as_deref_expr(
  --       cx, inputs, as_nid(cx, region),
  --       node.expr_type, node.annotations, node.span, privilege_map)
  --     return region
  --   end
  -- end

  return as_opaque_expr(
    cx,
    function(v1) return node { value = v1 } end,
    terralib.newlist({value}),
    privilege_map)
end

function flow_from_ast.expr_address_of(cx, node, privilege_map, init_only)
  local value = flow_from_ast.expr(cx, node.value, none)
  return as_opaque_expr(
    cx,
    function(v1) return node { value = v1 } end,
    terralib.newlist({value}),
    privilege_map)
end

function flow_from_ast.expr_import_ispace(cx, node, privilege_map, init_only)
  local value = flow_from_ast.expr(cx, node.value, reads)
  return as_opaque_expr(
    cx,
    function(v1) return node { value = v1 } end,
    terralib.newlist({value}),
    privilege_map)
end

function flow_from_ast.expr_import_region(cx, node, privilege_map, init_only)
  local ispace    = flow_from_ast.expr(cx, node.ispace,    reads)
  local value     = flow_from_ast.expr(cx, node.value,     reads)
  local field_ids = flow_from_ast.expr(cx, node.field_ids, reads)
  return as_opaque_expr(
    cx,
    function(v1, v2, v3) return node { ispace = v1, value = v2, field_ids = v3 } end,
    terralib.newlist({ispace, value, field_ids}),
    privilege_map)
end

function flow_from_ast.expr_import_partition(cx, node, privilege_map, init_only)
  local region = flow_from_ast.expr(cx, node.region, reads)
  local colors = flow_from_ast.expr(cx, node.colors, reads)
  local value  = flow_from_ast.expr(cx, node.value,  reads)
  return as_opaque_expr(
    cx,
    function(v1, v2, v3) return node { region = v1, colors = v2, value = v3 } end,
    terralib.newlist({region, colors, value}),
    privilege_map)
end

function flow_from_ast.expr(cx, node, privilege_map, init_only)
  if node:is(ast.typed.expr.ID) then
    return flow_from_ast.expr_id(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.Constant) then
    return flow_from_ast.expr_constant(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.Global) then
    return flow_from_ast.expr_global(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.Function) then
    return flow_from_ast.expr_function(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.FieldAccess) then
    return flow_from_ast.expr_field_access(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.IndexAccess) then
    return flow_from_ast.expr_index_access(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.MethodCall) then
    return flow_from_ast.expr_method_call(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.Call) then
    return flow_from_ast.expr_call(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.Cast) then
    return flow_from_ast.expr_cast(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.Ctor) then
    return flow_from_ast.expr_ctor(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.RawContext) then
    return flow_from_ast.expr_raw_context(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.RawFields) then
    return flow_from_ast.expr_raw_fields(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.RawFuture) then
    return flow_from_ast.expr_raw_future(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.RawPhysical) then
    return flow_from_ast.expr_raw_physical(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.RawRuntime) then
    return flow_from_ast.expr_raw_runtime(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.RawTask) then
    return flow_from_ast.expr_raw_task(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.RawValue) then
    return flow_from_ast.expr_raw_value(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.Isnull) then
    return flow_from_ast.expr_isnull(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.Null) then
    return flow_from_ast.expr_null(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.DynamicCast) then
    return flow_from_ast.expr_dynamic_cast(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.StaticCast) then
    return flow_from_ast.expr_static_cast(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.UnsafeCast) then
    return flow_from_ast.expr_unsafe_cast(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.Ispace) then
    return flow_from_ast.expr_ispace(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.Region) then
    return flow_from_ast.expr_region(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.Partition) then
    return flow_from_ast.expr_partition(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.PartitionEqual) then
    return flow_from_ast.expr_partition_equal(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.PartitionByField) then
    return flow_from_ast.expr_partition_by_field(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.PartitionByRestriction) then
    return flow_from_ast.expr_partition_by_restriction(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Image) then
    return flow_from_ast.expr_image(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.Preimage) then
    return flow_from_ast.expr_preimage(cx, node, privilege_map)

  elseif node:is(ast.typed.expr.CrossProduct) then
    return flow_from_ast.expr_cross_product(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.Advance) then
    return flow_from_ast.expr_advance(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.Adjust) then
    return flow_from_ast.expr_adjust(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.Arrive) then
    return flow_from_ast.expr_arrive(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.Await) then
    return flow_from_ast.expr_await(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.Copy) then
    return flow_from_ast.expr_copy(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.Fill) then
    return flow_from_ast.expr_fill(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.Acquire) then
    return flow_from_ast.expr_acquire(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.Release) then
    return flow_from_ast.expr_release(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.AttachHDF5) then
    return flow_from_ast.expr_attach_hdf5(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.DetachHDF5) then
    return flow_from_ast.expr_detach_hdf5(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.Unary) then
    return flow_from_ast.expr_unary(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.Binary) then
    return flow_from_ast.expr_binary(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.Deref) then
    return flow_from_ast.expr_deref(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.AddressOf) then
    return flow_from_ast.expr_address_of(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.ImportIspace) then
    return flow_from_ast.expr_import_ispace(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.ImportRegion) then
    return flow_from_ast.expr_import_region(cx, node, privilege_map, init_only)

  elseif node:is(ast.typed.expr.ImportPartition) then
    return flow_from_ast.expr_import_partition(cx, node, privilege_map, init_only)

  else
    assert(false, "unexpected node type " .. tostring(node.node_type))
  end
end

function flow_from_ast.block(cx, node)
  node.stats:map(
    function(stat) return flow_from_ast.stat(cx, stat) end)
end

function flow_from_ast.stat_if(cx, node)
  as_opaque_stat(cx, node)
end

function flow_from_ast.stat_while(cx, node)
  local loop_cx = cx:new_local_scope()
  local body_cx = cx:new_local_scope()

  local loop_block_privileges = analyze_privileges.stat(loop_cx, node)
  local loop_inner_privileges = index_privileges_by_region(
    privilege_summary(loop_cx, loop_block_privileges, false))
  local loop_outer_privileges = index_privileges_by_region(
    privilege_summary(cx, loop_block_privileges, true))
  for region_type, privilege_map in loop_inner_privileges:items() do
    preopen_region_tree(loop_cx, region_type, privilege_map)
  end
  local cond = flow_from_ast.expr(loop_cx, node.cond, reads)

  local body_block_privileges = analyze_privileges.block(body_cx, node.block)
  local body_inner_privileges = index_privileges_by_region(
    privilege_summary(body_cx, body_block_privileges, false))
  local body_outer_privileges = index_privileges_by_region(
    privilege_summary(loop_cx, body_block_privileges, true))
  for region_type, privilege_map in body_inner_privileges:items() do
    preopen_region_tree(body_cx, region_type, privilege_map)
  end
  flow_from_ast.block(body_cx, node.block)

  local body_inputs = terralib.newlist({cond})
  for region_type, privilege_map in body_outer_privileges:items() do
    local region_symbol = cx:region_symbol(region_type)
    local var_type = cx.tree:region_var_type(region_type)
    if not flow_region_tree.is_region(std.as_read(var_type)) then
      region_type = var_type
    end
    body_inputs:insert(open_region_tree(
                         loop_cx, region_type, region_symbol, privilege_map))
  end
  local body = as_while_body_stat(
    loop_cx, body_cx.graph, body_inputs, ast.default_annotations(), node.span)

  local loop_inputs = terralib.newlist()
  for region_type, privilege_map in loop_outer_privileges:items() do
    local region_symbol = cx:region_symbol(region_type)
    local var_type = cx.tree:region_var_type(region_type)
    if not flow_region_tree.is_region(std.as_read(var_type)) then
      region_type = var_type
    end
    loop_inputs:insert(open_region_tree(
                         cx, region_type, region_symbol, privilege_map))
  end
  as_while_loop_stat(cx, loop_cx.graph, loop_inputs, node.annotations, node.span)
end

function flow_from_ast.stat_for_num(cx, node)
  local inputs = node.values:map(
    function(value) return flow_from_ast.expr(cx, value, reads) end)

  local block_cx = cx:new_local_scope(node.symbol)
  block_cx:intern_region(node, node.symbol, node.symbol:gettype())

  local block_privileges = analyze_privileges.block(block_cx, node.block)
  local inner_privileges = index_privileges_by_region(
    privilege_summary(block_cx, block_privileges, false))
  local outer_privileges = index_privileges_by_region(
    privilege_summary(cx, block_privileges, true))
  for region_type, privilege_map in inner_privileges:items() do
    local var_type = cx.tree:region_var_type(region_type)
    if flow_region_tree.is_region(std.as_read(var_type)) then
      preopen_region_tree(block_cx, region_type, privilege_map)
    end
  end
  flow_from_ast.block(block_cx, node.block)

  do
    assert(#inputs <= 3)
    local i = 4
    for region_type, privilege_map in outer_privileges:items() do
      local region_symbol = cx:region_symbol(region_type)
      local var_type = cx.tree:region_var_type(region_type)
      if not flow_region_tree.is_region(std.as_read(var_type)) then
        region_type = var_type
      end
      inputs[i] = open_region_tree(
        cx, region_type, region_symbol, privilege_map)
      i = i + 1
    end
  end

  as_fornum_stat(
    cx, node.symbol, block_cx.graph, inputs, node.annotations, node.span)
end

function flow_from_ast.stat_for_list(cx, node)
  local value = flow_from_ast.expr(cx, node.value, name(node.value.expr_type))

  local block_cx = cx:new_local_scope(node.symbol)
  block_cx:intern_region(node, node.symbol, node.symbol:gettype())

  local block_privileges = analyze_privileges.block(block_cx, node.block)
  local inner_privileges = index_privileges_by_region(
    privilege_summary(block_cx, block_privileges, false))
  local outer_privileges = index_privileges_by_region(
    privilege_summary(cx, block_privileges, true))
  for region_type, privilege_map in inner_privileges:items() do
    local var_type = cx.tree:region_var_type(region_type)
    if flow_region_tree.is_region(std.as_read(var_type)) then
      preopen_region_tree(block_cx, region_type, privilege_map)
    end
  end
  flow_from_ast.block(block_cx, node.block)

  local inputs = terralib.newlist({value})
  for region_type, privilege_map in outer_privileges:items() do
    local region_symbol = cx:region_symbol(region_type)
    local var_type = cx.tree:region_var_type(region_type)
    if not flow_region_tree.is_region(std.as_read(var_type)) then
      region_type = var_type
    end
    inputs:insert(open_region_tree(
                    cx, region_type, region_symbol, privilege_map))
  end

  as_forlist_stat(
    cx, node.symbol, block_cx.graph, inputs, node.annotations, node.span)
end

function flow_from_ast.stat_repeat(cx, node)
  as_opaque_stat(cx, node)
end

function flow_from_ast.stat_must_epoch(cx, node)
  as_opaque_stat(cx, node)
end

function flow_from_ast.stat_block(cx, node)
  local block_cx = cx:new_local_scope()
  local block_privileges = analyze_privileges.block(block_cx, node.block)
  local inner_privileges = index_privileges_by_region(
    privilege_summary(block_cx, block_privileges, false))
  local outer_privileges = index_privileges_by_region(
    privilege_summary(cx, block_privileges, false))
  for region_type, privilege_map in inner_privileges:items() do
    local var_type = cx.tree:region_var_type(region_type)
    if flow_region_tree.is_region(std.as_read(var_type)) then
      preopen_region_tree(block_cx, region_type, privilege_map)
    end
  end
  flow_from_ast.block(block_cx, node.block)

  local inputs = terralib.newlist()
  for region_type, privilege_map in outer_privileges:items() do
    local region_symbol = cx:region_symbol(region_type)
    local var_type = cx.tree:region_var_type(region_type)
    if not flow_region_tree.is_region(std.as_read(var_type)) then
      region_type = var_type
    end
    inputs:insert(open_region_tree(
                    cx, region_type, region_symbol, privilege_map))
  end

  as_block_stat(
    cx, block_cx.graph, inputs, node.annotations, node.span)
end

function flow_from_ast.stat_var(cx, node)
  as_opaque_stat(cx, node)
end

function flow_from_ast.stat_var_unpack(cx, node)
  as_opaque_stat(cx, node)
end

function flow_from_ast.stat_return(cx, node)
  as_opaque_stat(cx, node)
end

function flow_from_ast.stat_break(cx, node)
  as_opaque_stat(cx, node)
end

function flow_from_ast.stat_assignment(cx, node)
  if not node.lhs:is(ast.typed.expr.ID) then
    as_opaque_stat(cx, node)
    return
  end

  local rhs = flow_from_ast.expr(cx, node.rhs, reads)
  local lhs = flow_from_ast.expr(cx, node.lhs, reads_writes)

  local inputs = terralib.newlist()
  inputs:insert(lhs)
  inputs:insert(rhs)

  as_assignment_stat(cx, inputs, node.annotations, node.span)
end

function flow_from_ast.stat_reduce(cx, node)
  local op = get_trivial_field_map(reduces(node.op))
  local rhs = flow_from_ast.expr(cx, node.rhs, reads)
  local lhs = flow_from_ast.expr(cx, node.lhs, op)

  local inputs = terralib.newlist()
  inputs:insert(lhs)
  inputs:insert(rhs)

  as_reduce_stat(cx, node.op, inputs, node.annotations, node.span)
end

function flow_from_ast.stat_expr(cx, node)
  local result = flow_from_ast.expr(cx, node.expr, name(node.expr.expr_type))
  for field_path, value in result:items() do
    local privilege, result_nid = unpack(value)
    if cx.graph:node_label(result_nid):is(flow.node.data.Scalar) then
      sequence_advance(cx, cx.graph:immediate_predecessor(result_nid))
    else
      sequence_advance(cx, result_nid)
    end
  end
end

function flow_from_ast.stat_raw_delete(cx, node)
  as_opaque_stat(cx, node)
end

function flow_from_ast.stat_fence(cx, node)
  as_opaque_stat(cx, node)
end

function flow_from_ast.stat_parallel_prefix(cx, node)
  -- TODO: Need a proper analysis
  as_opaque_stat(cx, node)
end

function flow_from_ast.stat(cx, node)
  if node:is(ast.typed.stat.If) then
    flow_from_ast.stat_if(cx, node)

  elseif node:is(ast.typed.stat.While) then
    flow_from_ast.stat_while(cx, node)

  elseif node:is(ast.typed.stat.ForNum) then
    flow_from_ast.stat_for_num(cx, node)

  elseif node:is(ast.typed.stat.ForList) then
    flow_from_ast.stat_for_list(cx, node)

  elseif node:is(ast.typed.stat.Repeat) then
    flow_from_ast.stat_repeat(cx, node)

  elseif node:is(ast.typed.stat.MustEpoch) then
    flow_from_ast.stat_must_epoch(cx, node)

  elseif node:is(ast.typed.stat.Block) then
    flow_from_ast.stat_block(cx, node)

  elseif node:is(ast.typed.stat.Var) then
    flow_from_ast.stat_var(cx, node)

  elseif node:is(ast.typed.stat.VarUnpack) then
    flow_from_ast.stat_var_unpack(cx, node)

  elseif node:is(ast.typed.stat.Return) then
    flow_from_ast.stat_return(cx, node)

  elseif node:is(ast.typed.stat.Break) then
    flow_from_ast.stat_break(cx, node)

  elseif node:is(ast.typed.stat.Assignment) then
    flow_from_ast.stat_assignment(cx, node)

  elseif node:is(ast.typed.stat.Reduce) then
    flow_from_ast.stat_reduce(cx, node)

  elseif node:is(ast.typed.stat.Expr) then
    flow_from_ast.stat_expr(cx, node)

  elseif node:is(ast.typed.stat.RawDelete) then
    flow_from_ast.stat_raw_delete(cx, node)

  elseif node:is(ast.typed.stat.Fence) then
    flow_from_ast.stat_fence(cx, node)

  elseif node:is(ast.typed.stat.ParallelPrefix) then
    flow_from_ast.stat_parallel_prefix(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function flow_from_ast.top_task(cx, node)
  if not node.body then return node end

  local task = node.prototype
  local cx = cx:new_task_scope(task:get_constraints(),
                               task:get_region_universe())
  for i, param in ipairs(node.params) do
    local param_type = std.rawref(&param.param_type)
    cx:intern_region(param, param.symbol, param_type)
  end
  analyze_regions.top_task(cx, node)
  analyze_privileges.block(cx, node.body)

  flow_from_ast.block(cx, node.body)
  return node { body = cx.graph }
end

function flow_from_ast.top(cx, node)
  if node:is(ast.typed.top.Task) then
    return flow_from_ast.top_task(cx, node)

  else
    return node
  end
end

function flow_from_ast.entry(node)
  local cx = context.new_global_scope()
  return flow_from_ast.top(cx, node)
end

flow_from_ast.pass_name = "flow_from_ast"

if std.config["flow"] then passes_hooks.add_optimization(15, flow_from_ast) end

return flow_from_ast
