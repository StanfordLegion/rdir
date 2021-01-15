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

-- Extract Node into a Subgraph containing the Node itself, all adjacent
-- (data) Nodes, and all edges connecting them.

local flow = require("regent/flow")

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

-- Extracts node `nid` into a subgraph.
-- Returns the subgraph and the node's new nid in the subgraph.
local function extract_subgraph(cx, nid)
  local subgraph_cx = cx:new_graph_scope(flow.empty_graph(cx.tree))

  local label = flow.node_label_deepcopy(cx.graph:node_label(nid))
  local compute_nid = subgraph_cx.graph:add_node(label)

  local inputs = cx.graph:incoming_edges(nid)
  for _, edge in ipairs(inputs) do
    if not edge.label:is(flow.edge.HappensBefore) then
      local input_label = cx.graph:node_label(edge.from_node)
      if input_label:is(flow.node.data.Scalar) then
        input_label = input_label { fresh = false }
      end
      local input_nid = subgraph_cx.graph:add_node(input_label)
      subgraph_cx.graph:add_edge(
        edge.label, input_nid, edge.from_port, compute_nid, edge.to_port)
    end
  end

  local outputs = cx.graph:outgoing_edges(nid)
  for _, edge in ipairs(outputs) do
    if not edge.label:is(flow.edge.HappensBefore) then
      local output_label = cx.graph:node_label(edge.to_node)
      if output_label:is(flow.node.data.Scalar) then
        output_label = output_label { fresh = false }
      end
      local output_nid = subgraph_cx.graph:add_node(output_label)
      subgraph_cx.graph:add_edge(
        edge.label, compute_nid, edge.from_port, output_nid, edge.to_port)
    end
  end

  return subgraph_cx.graph, compute_nid
end

local flow_extract_subgraph = {}

function flow_extract_subgraph.entry(graph, nid)
  assert(flow.is_graph(graph) and flow.is_valid_node(nid))
  local cx = context.new_global_scope():new_graph_scope(graph)
  return extract_subgraph(cx, nid)
end

return flow_extract_subgraph
