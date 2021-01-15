-- Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
-- Copyright (c) 2015-2021, Stanford University. All rights reserved.
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

-- Dataflow IR

local ast = require("regent/ast")
local data = require("common/data")

local flow = ast.make_factory("flow")

-- Dataflow Graph
local graph = setmetatable({}, {
    __index = function(t, k) error("graph has no field " .. tostring(k), 2) end
})
graph.__index = graph

function flow.empty_graph(region_tree)
  return setmetatable(
    {
      region_tree = region_tree,

      next_node = 1,
      -- node ID -> node label
      nodes = {},
      -- from node ID -> to node ID -> [(from port, to port, edge label)]
      edges = {},
      -- to node ID -> from node ID -> [(to port, from port, edge label)]
      backedges = {},
    }, graph)
end

function flow.is_graph(x)
  return getmetatable(x) == graph
end

function flow.null()
  return 0
end

function flow.is_null(node)
  return node == 0
end

function flow.is_valid_node(node)
  return node and node > 0
end

function flow.is_opaque_node(label)
  return label:is(flow.node.Opaque) or (
    label:is(flow.node.Task) and label.opaque) or
    -- FIXME: Depends on contents of subgraph.
    label:is(flow.node.ctrl)
end

function graph:has_node(node)
  assert(flow.is_valid_node(node))
  return self.nodes[node]
end

function graph:node_label(node)
  assert(self:has_node(node))
  return self.nodes[node]
end

function graph:set_node_label(node, label)
  assert(self:has_node(node))
  self.nodes[node] = label
end

function graph:node_result_port(node)
  return -1
end

function graph:node_sync_port(node)
  return -2
end

function graph:node_minimum_port(node)
  local label = self:node_label(node)
  if label:is(flow.node.Opaque) or
    label:is(flow.node.Task) or
    label:is(flow.node.Copy) or
    label:is(flow.node.Fill) or
    label:is(flow.node.Acquire) or
    label:is(flow.node.Release) or
    label:is(flow.node.AttachHDF5) or
    label:is(flow.node.DetachHDF5) or
    label:is(flow.node.Open) or
    label:is(flow.node.Close)
  then
    return 1
  elseif label:is(flow.node.ctrl.Block) then
    return 1
  elseif label:is(flow.node.ctrl.WhileLoop) then
    return 1
  elseif label:is(flow.node.ctrl.WhileBody) then
    return 2
  elseif label:is(flow.node.ctrl.ForNum) then
    return 4
  elseif label:is(flow.node.ctrl.ForList) then
    return 2
  elseif label:is(flow.node.ctrl.MustEpoch) then
    return 1
  elseif label:is(flow.node.data) or
    label:is(flow.node.Constant) or
    label:is(flow.node.Global) or
    label:is(flow.node.Function)
  then
    return 1
  else
    assert(false, "unexpected node type " .. tostring(label.node_type))
  end
end

function graph:node_available_port(node)
  local i = self:node_minimum_port(node)
  local inputs = self:incoming_edges_by_port(node)
  local outputs = self:outgoing_edges_by_port(node)
  while true do
    if (not rawget(inputs, i) or #inputs[i] == 0) and
      (not rawget(outputs, i) or #outputs[i] == 0)
    then
      return i
    end
    i = i + 1
  end
end

function graph:copy()
  local result = flow.empty_graph(self.region_tree)
  result.next_node = self.next_node
  for node, label in pairs(self.nodes) do
    result.nodes[node] = label
    result.edges[node] = {}
    result.backedges[node] = {}
  end
  for from, to_list in pairs(self.edges) do
    for to, edge_list in pairs(to_list) do
      for _, edge in pairs(edge_list) do
        result:add_edge(edge.label, from, edge.from_port, to, edge.to_port)
      end
    end
  end
  return result
end

function flow.node_label_deepcopy(label)
  local fields = {}
  for k, v in pairs(label) do
    if flow.is_graph(v) then
      fields[k] = v:deepcopy()
    end
  end
  return label(fields)
end

function graph:deepcopy()
  local result = self:copy()
  for node, label in pairs(result.nodes) do
    result.nodes[node] = flow.node_label_deepcopy(label)
  end
  return result
end

function graph:add_node(label)
  assert(label:is(flow.node))

  local node = self.next_node
  self.next_node = self.next_node + 1

  self.nodes[node] = label
  self.edges[node] = {}
  self.backedges[node] = {}
  return node
end

function graph:remove_node(node)
  assert(self:has_node(node))

  self.nodes[node] = nil
  self.edges[node] = nil
  self.backedges[node] = nil

  for _, to_list in pairs(self.edges) do
    if rawget(to_list, node) then
      to_list[node] = nil
    end
  end
  for _, from_list in pairs(self.backedges) do
    if rawget(from_list, node) then
      from_list[node] = nil
    end
  end
end

function graph:copy_node_edges(old_node, new_node)
  local inputs = self:incoming_edges(old_node)
  for _, edge in ipairs(inputs) do
    self:add_edge(
      edge.label, edge.from_node, edge.from_port, new_node, edge.to_port)
  end
  local outputs = self:outgoing_edges(old_node)
  for _, edge in ipairs(outputs) do
    self:add_edge(
      edge.label, new_node, edge.from_port, edge.to_node, edge.to_port)
  end
end

function graph:replace_node(old_node, new_node)
  self:copy_node_edges(old_node, new_node)
  self:remove_node(old_node)
end

function graph:add_edge(label, from_node, from_port, to_node, to_port)
  assert(label:is(flow.edge))
  assert(flow.is_valid_node(from_node) and flow.is_valid_node(to_node))
  assert(from_port and to_port)
  if not rawget(self.edges[from_node], to_node) then
    self.edges[from_node][to_node] = terralib.newlist()
  end
  self.edges[from_node][to_node]:insert(
    {
      from_port = from_port,
      to_port = to_port,
      label = label,
  })
  if not rawget(self.backedges[to_node], from_node) then
    self.backedges[to_node][from_node] = terralib.newlist()
  end
  self.backedges[to_node][from_node]:insert(
    {
      to_port = to_port,
      from_port = from_port,
      label = label,
  })
end

function graph:remove_edge(from_node, from_port, to_node, to_port)
  if rawget(self.edges, from_node) and rawget(self.edges[from_node], to_node)
  then
    self.edges[from_node][to_node] = data.filter(
      function(edge)
        return edge.from_port ~= from_port or edge.to_port ~= to_port
      end,
      self.edges[from_node][to_node])
  end
  if rawget(self.backedges, to_node) and
    rawget(self.backedges[to_node], from_node)
  then
    self.backedges[to_node][from_node] = data.filter(
      function(edge)
        return edge.from_port ~= from_port or edge.to_port ~= to_port
      end,
      self.backedges[to_node][from_node])
  end
end

function graph:set_edge_label(label, from_node, from_port, to_node, to_port)
  if rawget(self.edges, from_node) and rawget(self.edges[from_node], to_node)
  then
    for _, edge in ipairs(self.edges[from_node][to_node]) do
      if edge.from_port == from_port or edge.to_port == to_port then
        edge.label = label
      end
    end
  end
  if rawget(self.backedges, to_node) and
    rawget(self.backedges[to_node], from_node)
  then
    for _, edge in ipairs(self.backedges[to_node][from_node]) do
      if edge.from_port == from_port or edge.to_port == to_port then
        edge.label = label
      end
    end
  end
end

function graph:copy_edges(node, old_node, new_node, new_port)
  for _, edge in ipairs(self:incoming_edges(node)) do
    if edge.from_node == old_node then
      self:add_edge(
        edge.label, new_node, edge.from_port,
        edge.to_node, new_port or edge.to_port)
    end
  end
  for _, edge in ipairs(self:outgoing_edges(node)) do
    if edge.to_node == old_node then
      self:add_edge(
        edge.label, edge.from_node, new_port or edge.from_port,
        new_node, edge.to_port)
    end
  end
end

function graph:replace_edges(node, old_node, new_node)
  for _, edge in ipairs(self:incoming_edges(node)) do
    if edge.from_node == old_node then
      self:remove_edge(
        edge.from_node, edge.from_port, edge.to_node, edge.to_port)
      self:add_edge(
        edge.label, new_node, edge.from_port, edge.to_node, edge.to_port)
    end
  end
  for _, edge in ipairs(self:outgoing_edges(node)) do
    if edge.to_node == old_node then
      self:remove_edge(
        edge.from_node, edge.from_port, edge.to_node, edge.to_port)
      self:add_edge(
        edge.label, edge.from_node, edge.from_port, new_node, edge.to_port)
    end
  end
end

function graph:traverse_nodes(fn)
  for node, label in pairs(self.nodes) do
    local result = {fn(node, label)}
    if result[1] ~= nil then return unpack(result) end
  end
end

function graph:find_node(fn)
  return self:traverse_nodes(
    function(node, label)
      if fn(node, label) then
        return node, label
      end
    end)
end

function graph:filter_nodes(fn)
  local result = terralib.newlist()
  self:traverse_nodes(
    function(node, label)
      if fn(node, label) then
        result:insert(node)
      end
    end)
  return result
end

function graph:any_nodes(fn)
  return self:traverse_nodes(
    function(node, label) return fn(node, label) or nil end) or false
end

function graph:traverse_nodes_recursive(fn, continue_recursion)
  for node, label in pairs(self.nodes) do
    if not continue_recursion or continue_recursion(self, node, label) then
      for k, v in pairs(label) do
        if flow.is_graph(v) then
          local result = {v:traverse_nodes_recursive(fn, continue_recursion)}
          if result[1] ~= nil then return unpack(result) end
        end
      end
    end
    local result = {fn(self, node, label)}
    if result[1] ~= nil then return unpack(result) end
  end
end

function graph:find_node_recursive(fn)
  return self:traverse_nodes_recursive(
    function(graph, node, label)
      if fn(graph, node, label) then
        return graph, node, label
      end
    end)
end

function graph:map_nodes_recursive(fn)
  for node, label in pairs(self.nodes) do
    for k, v in pairs(label) do
      if flow.is_graph(v) then
        v:map_nodes_recursive(fn)
      end
    end
    local new_label = fn(self, node, label)
    assert(new_label:is(flow.node))
    self.nodes[node] = new_label
  end
end

function graph:traverse_edges(fn)
  for from, to_list in pairs(self.edges) do
    for to, edge_list in pairs(to_list) do
      for _, edge in pairs(edge_list) do
        fn(from, edge.from_port, self:node_label(from),
           to, edge.to_port, self:node_label(to),
           edge.label)
      end
    end
  end
end

local function pack_edge(from_node, to_node, edge)
  return {
    from_node = from_node,
    from_port = edge.from_port,
    to_node = to_node,
    to_port = edge.to_port,
    label = edge.label,
  }
end

function graph:map_edges(fn)
  for from, to_list in pairs(self.edges) do
    for to, edge_list in pairs(to_list) do
      for _, edge in pairs(edge_list) do
        local result = fn(pack_edge(from, to, edge))
        if result ~= nil then
          assert(result:is(flow.edge))
          self:set_edge_label(
            result, from, edge.from_port, to, edge.to_port)
        end
      end
    end
  end
end

function graph:traverse_incoming_edges(fn, node)
  assert(self:has_node(node))
  if rawget(self.backedges, node) then
    for from_node, edges in pairs(self.backedges[node]) do
      for _, edge in pairs(edges) do
        fn(from_node, node, edge)
      end
    end
  end
end

function graph:incoming_edges(node)
  local result = terralib.newlist()
  self:traverse_incoming_edges(
    function(from_node, to_node, edge)
      result:insert(pack_edge(from_node, to_node, edge))
    end,
    node)
  return result
end

function graph:filter_incoming_edges(fn, node)
  local result = terralib.newlist()
  self:traverse_incoming_edges(
    function(from_node, to_node, edge)
      if fn(pack_edge(from_node, to_node, edge)) then
        result:insert(pack_edge(from_node, to_node, edge))
      end
    end,
    node)
  return result
end

function graph:copy_incoming_edges(fn, old_node, new_node, delete_old)
  local edges = self:filter_incoming_edges(fn, old_node)
  for _, edge in ipairs(edges) do
    self:add_edge(
      edge.label, edge.from_node, edge.from_port, new_node, edge.to_port)
    if delete_old then
      self:remove_edge(
        edge.from_node, edge.from_port, edge.to_node, edge.to_port)
    end
  end
end

function graph:remove_incoming_edges(fn, node)
  local edges = self:filter_incoming_edges(fn, node)
  for _, edge in ipairs(edges) do
    self:remove_edge(
      edge.from_node, edge.from_port, edge.to_node, edge.to_port)
  end
end

function graph:incoming_edges_by_port(node)
  local result = {}
  self:traverse_incoming_edges(
    function(from_node, to_node, edge)
      if not rawget(result, edge.to_port) then
        result[edge.to_port] = terralib.newlist()
      end
      result[edge.to_port]:insert(pack_edge(from_node, to_node, edge))
    end,
    node)
  return result
end

function graph:traverse_outgoing_edges(fn, node)
  assert(self:has_node(node))
  if rawget(self.edges, node) then
    for to_node, edges in pairs(self.edges[node]) do
      for _, edge in pairs(edges) do
        fn(node, to_node, edge)
      end
    end
  end
end

function graph:outgoing_edges(node)
  local result = terralib.newlist()
  self:traverse_outgoing_edges(
    function(from_node, to_node, edge)
      result:insert(pack_edge(from_node, to_node, edge))
    end,
    node)
  return result
end

function graph:filter_outgoing_edges(fn, node)
  local result = terralib.newlist()
  self:traverse_outgoing_edges(
    function(from_node, to_node, edge)
      if fn(pack_edge(from_node, to_node, edge)) then
        result:insert(pack_edge(from_node, to_node, edge))
      end
    end,
    node)
  return result
end

function graph:copy_outgoing_edges(fn, old_node, new_node, delete_old)
  local edges = self:filter_outgoing_edges(fn, old_node)
  for _, edge in ipairs(edges) do
    self:add_edge(
      edge.label, new_node, edge.from_port, edge.to_node, edge.to_port)
    if delete_old then
      self:remove_edge(
        edge.from_node, edge.from_port, edge.to_node, edge.to_port)
    end
  end
end

function graph:remove_outgoing_edges(fn, node)
  local edges = self:filter_outgoing_edges(fn, node)
  for _, edge in ipairs(edges) do
    self:remove_edge(
      edge.from_node, edge.from_port, edge.to_node, edge.to_port)
  end
end

function graph:outgoing_edges_by_port(node)
  local result = {}
  self:traverse_outgoing_edges(
    function(from_node, to_node, edge)
      if not rawget(result, edge.from_port) then
        result[edge.from_port] = terralib.newlist()
      end
      result[edge.from_port]:insert(pack_edge(from_node, to_node, edge))
    end,
    node)
  return result
end

function graph:traverse_immediate_predecessors(fn, node)
  assert(self:has_node(node))
  for from_node, edges in pairs(self.backedges[node]) do
    if #edges > 0 then
      local from_node_label = self:node_label(from_node)
      local result = {fn(from_node, from_node_label)}
      if result[1] ~= nil then return unpack(result) end
    end
  end
end

function graph:traverse_immediate_successors(fn, node)
  assert(self:has_node(node))
  for to_node, edges in pairs(self.edges[node]) do
    if #edges > 0 then
      local to_node_label = self:node_label(to_node)
      local result = {fn(to_node, to_node_label)}
      if result[1] ~= nil then return unpack(result) end
    end
  end
end

function graph:immediate_predecessor(node)
  local result
  self:traverse_immediate_predecessors(
    function(pred) assert(not result); result = pred end,
    node)
  assert(result)
  return result
end

function graph:immediate_successor(node)
  local result
  self:traverse_immediate_successors(
    function(succ) assert(not result); result = succ end,
    node)
  assert(result)
  return result
end

function graph:immediate_predecessors(node)
  local result = terralib.newlist()
  self:traverse_immediate_predecessors(
    function(pred) result:insert(pred) end,
    node)
  return result
end

function graph:immediate_successors(node)
  local result = terralib.newlist()
  self:traverse_immediate_successors(
    function(succ) result:insert(succ) end,
    node)
  return result
end

function graph:find_immediate_predecessor(fn, node)
  return self:traverse_immediate_predecessors(
    function(...) if fn(...) then return ... end end,
    node)
end

function graph:find_immediate_successor(fn, node)
  return self:traverse_immediate_successors(
    function(...) if fn(...) then return ... end end,
    node)
end

function graph:filter_immediate_predecessors(fn, node)
  local result = terralib.newlist()
  self:traverse_immediate_predecessors(
    function(pred, label) if fn(pred, label) then result:insert(pred) end end,
    node)
  return result
end

function graph:filter_immediate_predecessors_by_edges(fn, node)
  assert(self:has_node(node))
  local result = terralib.newlist()
  if rawget(self.backedges, node) then
    for from_node, edges in pairs(self.backedges[node]) do
      for _, edge in pairs(edges) do
        local label = self:node_label(from_node)
        if fn(pack_edge(from_node, node, edge), label) then
          result:insert(from_node)
          break
        end
      end
    end
  end
  return result
end

function graph:filter_immediate_successors(fn, node)
  local result = terralib.newlist()
  self:traverse_immediate_successors(
    function(succ, label) if fn(succ, label) then result:insert(succ) end end,
    node)
  return result
end

function graph:filter_immediate_successors_by_edges(fn, node)
  assert(self:has_node(node))
  local result = terralib.newlist()
  if rawget(self.edges, node) then
    for to_node, edges in pairs(self.edges[node]) do
      for _, edge in pairs(edges) do
        local label = self:node_label(to_node)
        if fn(pack_edge(node, to_node, edge), label) then
          result:insert(to_node)
          break
        end
      end
    end
  end
  return result
end

function graph:incoming_read_set(node)
  return self:filter_immediate_predecessors_by_edges(
    function(edge) return edge.label:is(flow.edge.Read) end, node)
end

function graph:incoming_name_set(node)
  return self:filter_immediate_predecessors_by_edges(
    function(edge) return edge.label:is(flow.edge.Name) end, node)
end

function graph:incoming_write_set(node)
  return self:filter_immediate_predecessors_by_edges(
    function(edge) return edge.label:is(flow.edge.Write) end, node)
end

function graph:incoming_mutate_set(node)
  return self:filter_immediate_predecessors_by_edges(
    function(edge)
      return edge.label:is(flow.edge.Write) or edge.label:is(flow.edge.Reduce)
    end,
    node)
end

function graph:incoming_use_set(node)
  return self:filter_immediate_predecessors_by_edges(
    function(edge)
      return edge.label:is(flow.edge.Reduce) or edge.label:is(flow.edge.Arrive)
    end,
    node)
end

function graph:outgoing_read_set(node)
  return self:filter_immediate_successors_by_edges(
    function(edge) return edge.label:is(flow.edge.Read) end, node)
end

function graph:outgoing_name_set(node)
  return self:filter_immediate_successors_by_edges(
    function(edge) return edge.label:is(flow.edge.Name) end, node)
end

function graph:outgoing_use_set(node)
  return self:filter_immediate_successors_by_edges(
    function(edge)
      return edge.label:is(flow.edge.None) or edge.label:is(flow.edge.Read) or
        edge.label:is(flow.edge.Await)
    end,
    node)
end

function graph:outgoing_write_set(node)
  return self:filter_immediate_successors_by_edges(
    function(edge) return edge.label:is(flow.edge.Write) end, node)
end

function graph:node_result_is_used(node)
  local outputs = self:outgoing_edges_by_port(node)[self:node_result_port(node)]
  if outputs then
    for _, edge in ipairs(outputs) do
      if #self:outgoing_use_set(edge.to_node) > 0 or #self:incoming_use_set(edge.to_node) > 0 then
        return true
      end
    end
  end
  return false
end

local function dfs(graph, node, edge_filter, stop_predicate, visited)
  if stop_predicate(node) then
    return true
  end

  if rawget(visited, node) then
    return false
  end
  visited[node] = true

  for to_node, edges in pairs(graph.edges[node]) do
    local accept = false
    if edge_filter then
      for _, edge in ipairs(edges) do
        if edge_filter(pack_edge(node, to_node, edge)) then
          accept = true
          break
        end
      end
    else
      accept = true
    end

    if accept and dfs(graph, to_node, edge_filter, stop_predicate, visited) then
      return true
    end
  end
  return false
end

function graph:reachable(src_node, dst_node, edge_filter)
  return dfs(
    self, src_node, edge_filter, function(node) return node == dst_node end, {})
end

function graph:between_nodes(src_node, dst_node)
  local result = terralib.newlist()
  dfs(
    self, src_node, false, function(node)
      if node ~= src_node and node ~= dst_node and
        dfs(self, node, false, function(other) return other == dst_node end, {})
      then
        result:insert(node)
      end
    end, {})
  return result
end

local function toposort_node(graph, node, visited, path, sort)
  if rawget(path, node) then
    error("cycle in toposort at " .. tostring(node))
  end
  if not rawget(visited, node) then
    path[node] = true
    for _, child in pairs(graph:immediate_successors(node)) do
      toposort_node(graph, child, visited, path, sort)
    end
    path[node] = false
    visited[node] = true
    sort:insert(node)
  end
end

local function inverse_toposort_node(graph, node, visited, path, sort)
  if rawget(path, node) then
    error("cycle in inverse_toposort at " .. tostring(node))
  end
  if not rawget(visited, node) then
    path[node] = true
    for _, child in pairs(graph:immediate_predecessors(node)) do
      inverse_toposort_node(graph, child, visited, path, sort)
    end
    path[node] = false
    visited[node] = true
    sort:insert(node)
  end
end

local function reverse(list)
  local result = terralib.newlist()
  for i = #list, 1, -1 do
    result:insert(list[i])
  end
  return result
end

function graph:toposort()
  local visited = {}
  local sort = terralib.newlist()
  self:traverse_nodes(
    function(node)
      toposort_node(self, node, visited, {}, sort)
  end)
  return reverse(sort)
end

function graph:inverse_toposort()
  local visited = {}
  local sort = terralib.newlist()
  self:traverse_nodes(
    function(node)
      inverse_toposort_node(self, node, visited, {}, sort)
  end)
  return reverse(sort)
end

local function filter_toposort_node(graph, node, visited, path, sort,
                                    node_filter, edge_filter)
  if rawget(path, node) then
    error("cycle in toposort at " .. tostring(node))
  end
  if not rawget(visited, node) then
    path[node] = true

    local children
    if edge_filter then
      children = graph:filter_immediate_successors_by_edges(edge_filter, node)
    else
      children = graph:immediate_successors(node)
    end

    for _, child in pairs(children) do
      filter_toposort_node(graph, child, visited, path, sort,
                           node_filter, edge_filter)
    end

    path[node] = false
    visited[node] = true
    if node_filter(node, graph:node_label(node)) then sort:insert(node) end
  end
end

local function filter_inverse_toposort_node(graph, node, visited, path, sort,
                                            node_filter, edge_filter)
  if rawget(path, node) then
    error("cycle in inverse_toposort at " .. tostring(node))
  end
  if not rawget(visited, node) then
    path[node] = true

    local children
    if edge_filter then
      children = graph:filter_immediate_predecessors_by_edges(edge_filter, node)
    else
      children = graph:immediate_predecessors(node)
    end

    for _, child in pairs(children) do
      filter_inverse_toposort_node(graph, child, visited, path, sort,
                                   node_filter, edge_filter)
    end

    path[node] = false
    visited[node] = true
    if node_filter(node, graph:node_label(node)) then sort:insert(node) end
  end
end

function graph:filter_toposort(node, node_filter, edge_filter)
  local visited = {}
  local sort = terralib.newlist()
  if node then
    filter_toposort_node(
      self, node, visited, {}, sort, node_filter, edge_filter)
  else
    self:traverse_nodes(
      function(node)
        filter_toposort_node(
          self, node, visited, {}, sort, node_filter, edge_filter)
      end)
  end
  return reverse(sort)
end

function graph:filter_inverse_toposort(node, node_filter, edge_filter)
  local visited = {}
  local sort = terralib.newlist()
  if node then
    filter_inverse_toposort_node(
      self, node, visited, {}, sort, node_filter, edge_filter)
  else
    self:traverse_nodes(
      function(node)
        filter_inverse_toposort_node(
          self, node, visited, {}, sort, node_filter, edge_filter)
      end)
  end
  return reverse(sort)
end

function graph:printpretty(ids, types, metadata)
  print("digraph {")
  print("rankdir = LR;")
  print("node [ margin = \"0.055,0.0275\" ];")
  self:traverse_nodes(function(i, node)
    local label = tostring(node:type()):gsub("[^.]+[.]", ""):lower()
    if node:is(flow.node.data) or
      node:is(flow.node.Constant) or node:is(flow.node.Global) or node:is(flow.node.Function)
    then
      local name = tostring(node.value.value):gsub("\n", "\\n")
      if terralib.isfunction(node.value.value) then
        name = tostring(node.value.value.name)
      end
      label = label .. " " .. tostring(name)
      if node:is(flow.node.data) then
        if types then label = label .. " " .. tostring(node.region_type) end
        label = label .. " " .. tostring(node.field_path)
      end
    elseif (node:is(flow.node.Copy) or node:is(flow.node.Reduce)) and node.op then
      label = label .. " " .. tostring(node.op)
    end
    if ids then label = label .. " " .. tostring(i) end
    if metadata then label = label .. " " .. tostring(metadata[i]) end
    local shape
    if node:is(flow.node.Opaque) or
      node:is(flow.node.Binary) or
      node:is(flow.node.Cast) or
      node:is(flow.node.IndexAccess) or
      node:is(flow.node.Deref) or
      node:is(flow.node.Advance) or
      node:is(flow.node.Assignment) or node:is(flow.node.Reduce) or
      node:is(flow.node.Task) or
      node:is(flow.node.Copy) or node:is(flow.node.Fill) or
      node:is(flow.node.Acquire) or node:is(flow.node.Release) or
      node:is(flow.node.AttachHDF5) or node:is(flow.node.DetachHDF5) or
      node:is(flow.node.ctrl)
    then
      shape = "rectangle"
    elseif node:is(flow.node.Open) then
      shape = "polygon, sides = 3, orientation = 90"
    elseif node:is(flow.node.Close) then
      shape = "polygon, sides = 3, orientation = 270"
    elseif node:is(flow.node.data) or
      node:is(flow.node.Constant) or node:is(flow.node.Global) or node:is(flow.node.Function)
    then
      shape = "ellipse"
    else
      error("unexpected node type " .. tostring(node.node_type))
    end
    print(tostring(i) .. " [ label = \"" .. label .. "\", shape = " .. shape .. " ];")
  end)
  self:traverse_edges(
    function(from, from_port, from_label, to, to_port, to_label, edge)
      local label = tostring(edge:type()):gsub("[^.]+[.]", ""):lower()
      if edge:is(flow.edge.Reduce) then
        label = label .. " " .. tostring(edge.op)
      end
      if edge:is(flow.edge.None) or edge:is(flow.edge.Read) or
        edge:is(flow.edge.Discard) or edge:is(flow.edge.Write) or
        edge:is(flow.edge.Reduce)
      then
        if not edge.coherence:is(flow.coherence_kind.Exclusive) then
          label = label .. " " .. tostring(edge.coherence:type()):gsub("[^.]+[.]", ""):lower()
        end
        if not edge.flag:is(flow.flag_kind.NoFlag) then
          label = label .. " " .. tostring(edge.flag:type()):gsub("[^.]+[.]", ""):lower()
        end
      end
      if ids then label = tostring(from_port) .. " " .. label .. " " .. tostring(to_port) end
      local style = "solid"
      if edge:is(flow.edge.HappensBefore) then
        style = "dotted"
      end
      print(tostring(from) .. " -> " .. tostring(to) ..
              " [ label = \"" .. label .. "\", style = \"" .. style .. "\" ];")
  end)
  print("}")
end

-- Dataflow Graph: Nodes
flow:inner("node")

-- Compute
flow.node:leaf("Opaque", {"action"})

flow.node:leaf("Binary", {"op", "expr_type", "annotations", "span"})
flow.node:leaf("Cast", {"expr_type", "annotations", "span"})
flow.node:leaf("IndexAccess", {"expr_type", "annotations", "span"})
flow.node:leaf("Deref", {"expr_type", "annotations", "span"})
flow.node:leaf("Advance", {"expr_type", "annotations", "span"})
flow.node:leaf("Assignment", {"annotations", "span"})
flow.node:leaf("Reduce", {"op", "annotations", "span"})
flow.node:leaf("Task", {"opaque", "replicable", "expr_type", "annotations", "span"})
flow.node:leaf("Copy", {"src_field_paths", "dst_field_paths",
                        "op", "annotations", "span"})
flow.node:leaf("Fill", {"dst_field_paths", "annotations", "span"})
flow.node:leaf("Acquire", {"field_paths", "annotations", "span"})
flow.node:leaf("Release", {"field_paths", "annotations", "span"})
flow.node:leaf("AttachHDF5", {"field_paths", "annotations", "span"})
flow.node:leaf("DetachHDF5", {"field_paths", "annotations", "span"})

flow.node:leaf("Open", {})
flow.node:leaf("Close", {})

-- Control
flow.node:inner("ctrl", {"block", "annotations", "span"})
flow.node.ctrl:leaf("Block", {})
flow.node.ctrl:leaf("WhileLoop", {})
flow.node.ctrl:leaf("WhileBody", {})
flow.node.ctrl:leaf("ForNum", {"symbol"})
flow.node.ctrl:leaf("ForList", {"symbol"})
flow.node.ctrl:leaf("MustEpoch", {})

-- Data
flow.node:inner("data", {"value", "region_type", "field_path"})
flow.node.data:leaf("Region", {})
flow.node.data:leaf("Partition", {})
flow.node.data:leaf("CrossProduct", {})
flow.node.data:leaf("List", {})
flow.node.data:leaf("Scalar", {"fresh"})
flow.node:leaf("Constant", {"value"})
flow.node:leaf("Global", {"value"})
flow.node:leaf("Function", {"value"})

-- Dataflow Graph: Edges
flow:inner("coherence_kind")
flow.coherence_kind:leaf("Exclusive")
flow.coherence_kind:leaf("Atomic")
flow.coherence_kind:leaf("Simultaneous")
flow.coherence_kind:leaf("Relaxed")

flow:inner("flag_kind", {})
flow.flag_kind:leaf("NoFlag")
flow.flag_kind:leaf("NoAccessFlag")

flow:inner("edge")

flow.edge:leaf("HappensBefore", {})
flow.edge:leaf("Name", {})

flow.edge:leaf("None", {"coherence", "flag"})
flow.edge:leaf("Read", {"coherence", "flag"})
flow.edge:leaf("Discard", {"coherence", "flag"})
flow.edge:leaf("Write", {"coherence", "flag"})
flow.edge:leaf("Reduce", {"coherence", "flag", "op"})

flow.edge:leaf("Await", {})
flow.edge:leaf("Arrive", {})

function flow.default_coherence()
  return flow.coherence_kind.Exclusive {}
end

function flow.default_flag()
  return flow.flag_kind.NoFlag {}
end

function flow.default_mode()
  return {
    coherence = flow.coherence_kind.Exclusive {},
    flag = flow.flag_kind.NoFlag {},
  }
end

return flow
