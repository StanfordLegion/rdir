-- Copyright (c) 2015-2021, Stanford University. All rights reserved.
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

-- Summarize Privileges used in Subgraph of Node

local data = require("common/data")
local flow = require("regent/flow")
local std = require("regent/std")

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

local function privilege_kind(label)
  if label:is(flow.edge.None) then
    return "none"
  elseif label:is(flow.edge.Read) then
    return "reads"
  elseif label:is(flow.edge.Write) then
    return "reads_writes"
  elseif label:is(flow.edge.Reduce) then
    return tostring(std.reduces(label.op))
  end
end

local function coherence_kind(label)
  if label:is(flow.edge.None) or label:is(flow.edge.Read) or
    label:is(flow.edge.Write) or label:is(flow.edge.Reduce)
  then
    if label.coherence:is(flow.coherence_kind.Exclusive) then
      return "exclusive"
    elseif label.coherence:is(flow.coherence_kind.Atomic) then
      return "atomic"
    elseif label.coherence:is(flow.coherence_kind.Simultaneous) then
      return "simultaneous"
    elseif label.coherence:is(flow.coherence_kind.Relaxed) then
      return "relaxed"
    else
      assert(false)
    end
  end
end

local function flag_kind(label)
  if label:is(flow.edge.None) or label:is(flow.edge.Read) or
    label:is(flow.edge.Write) or label:is(flow.edge.Reduce)
  then
    if label.flag:is(flow.flag_kind.NoFlag) then
      return "no_flag"
    elseif label.flag:is(flow.flag_kind.NoAccessFlag) then
      return "no_access_flag"
    else
      assert(false)
    end
  end
end

local function coherence_kind_label(kind)
  if kind == "exclusive" then
    return flow.coherence_kind.Exclusive {}
  elseif kind == "atomic" then
    return flow.coherence_kind.Atomic {}
  elseif kind == "simultaneous" then
    return flow.coherence_kind.Simultaneous {}
  elseif kind == "relaxed" then
    return flow.coherence_kind.Relaxed {}
  else
    assert(false)
  end
end

local function flag_kind_label(kind)
  if not kind or kind == "no_flag" then
    return flow.flag_kind.NoFlag {}
  elseif kind == "no_access_flag" then
    return flow.flag_kind.NoAccessFlag {}
  else
    assert(false)
  end
end

local function summarize_subgraph(cx, nid, mapping)
  local label = cx.graph:node_label(nid)
  assert(label:is(flow.node.ctrl))

  local block_cx = cx:new_graph_scope(label.block)
  local usage_modes = data.new_recursive_map(1)
  local labels = data.new_recursive_map(1)
  block_cx.graph:traverse_nodes(
    function(nid, label)
      if label:is(flow.node.data) then
        local region_type = label.region_type
        local field_path = label.field_path

        if mapping[region_type] == false then
          return
        end

        if mapping[region_type] then
          region_type = mapping[region_type]
          local nid1, label1 = block_cx.graph:find_node(
            function(nid, label)
              return label:is(flow.node.data) and
                label.region_type == region_type and
                label.field_path == field_path
            end)
          assert(label1)
          label = label1
        end

        local inputs = block_cx.graph:incoming_edges(nid)
        local outputs = block_cx.graph:outgoing_edges(nid)
        for _, edges in ipairs({inputs, outputs}) do
          for _, edge in ipairs(edges) do
            local privilege = privilege_kind(edge.label)
            local coherence = coherence_kind(edge.label)
            local flag = flag_kind(edge.label)
            if privilege then
              assert(coherence and flag)
              local old_privilege, old_coherence, old_flag = usage_modes[region_type][field_path] and
                unpack(usage_modes[region_type][field_path])
              local new_privilege = std.meet_privilege(privilege, old_privilege)
              local new_coherence = std.meet_coherence(coherence, old_coherence)
              local new_flag = std.meet_flag(flag, old_flag)
              usage_modes[region_type][field_path] =
                data.newtuple(new_privilege, new_coherence, new_flag)

              if not labels[region_type][field_path] then
                labels[region_type][field_path] = label
              end
            end
          end
        end
      end
    end)

  for region_type, modes in usage_modes:items() do
    local port
    for field_path, mode in modes:items() do
      local privilege, coherence, flag = unpack(mode)
      local label = labels[region_type][field_path]

      local read_edge_label
      if privilege == "none" then
        read_edge_label = flow.edge.None {
          coherence = coherence_kind_label(coherence),
          flag = flag_kind_label(flag),
        }
      elseif privilege == "reads" or privilege == "reads_writes" then
        read_edge_label = flow.edge.Read {
          coherence = coherence_kind_label(coherence),
          flag = flag_kind_label(flag),
        }
      elseif std.is_reduction_op(privilege) then
        -- Skip
      else
        assert(false)
      end
      if read_edge_label then
        local read_nid = cx.graph:add_node(label)
        port = port or cx.graph:node_available_port(nid)
        cx.graph:add_edge(
          read_edge_label, read_nid, cx.graph:node_result_port(read_nid),
          nid, port)
      end

      local write_edge_label
      if privilege == "none" or privilege == "reads" then
        -- Skip
      elseif privilege == "reads_writes" then
        write_edge_label = flow.edge.Write {
          coherence = coherence_kind_label(coherence),
          flag = flag_kind_label(flag),
        }
      elseif std.is_reduction_op(privilege) then
        write_edge_label = flow.edge.Reduce {
          coherence = coherence_kind_label(coherence),
          flag = flag_kind_label(flag),
          op = std.get_reduction_op(privilege),
        }
      else
        assert(false)
      end

      if write_edge_label then
        local write_nid = cx.graph:add_node(label)
        port = port or cx.graph:node_available_port(nid)
        cx.graph:add_edge(
          write_edge_label, nid, port,
          write_nid, cx.graph:node_available_port(write_nid))
      end
    end
  end
end

local flow_summarize_subgraph = {}

function flow_summarize_subgraph.entry(graph, nid, mapping)
  assert(flow.is_graph(graph) and flow.is_valid_node(nid) and mapping)
  local cx = context.new_global_scope():new_graph_scope(graph)
  return summarize_subgraph(cx, nid, mapping)
end

return flow_summarize_subgraph
