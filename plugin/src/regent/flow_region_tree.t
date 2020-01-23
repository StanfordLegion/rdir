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

local std = require("regent/std")

local flow_region_tree = {}

-- For the purposes of analysis, consider partitions and lists (of
-- regions) to be regions as well.
function flow_region_tree.is_region(region_type)
  return std.is_region(region_type) or std.is_partition(region_type) or
    std.is_cross_product(region_type) or std.is_list_of_regions(region_type) or
    std.is_list_of_partitions(region_type)
end

-- Region Tree

local region_tree = setmetatable({}, { __index = function(t, k) error("region tree has no field " .. tostring(k), 2) end})
region_tree.__index = region_tree

function flow_region_tree.new_region_tree(constraints, region_universe)
  -- Copy region_universe to allow safe modifications.
  local initial_universe = region_universe:copy()
  return setmetatable({
      -- Region tree structure.
      constraints = constraints,
      region_universe = initial_universe,
      interned_scalars = {},

      -- Region identity and indexing.
      region_var_types = {},
      region_annotation_sets = {},
      region_spans = {},
      region_indices = {},
      region_is_point = {},
      region_point_partitions = {},
  }, region_tree)
end

function region_tree:has_region(region_type)
  return self:has_region_span(region_type)
end

function region_tree:region_var_type(region_type)
  assert(flow_region_tree.is_region(region_type))
  assert(rawget(self.region_var_types, region_type))
  return self.region_var_types[region_type]
end

function region_tree:region_annotations(region_type)
  assert(flow_region_tree.is_region(region_type))
  assert(rawget(self.region_annotation_sets, region_type))
  return self.region_annotation_sets[region_type]
end

function region_tree:has_region_span(region_type)
  assert(flow_region_tree.is_region(region_type))
  return rawget(self.region_spans, region_type)
end

function region_tree:region_span(region_type)
  assert(flow_region_tree.is_region(region_type))
  assert(rawget(self.region_spans, region_type))
  return self.region_spans[region_type]
end

function region_tree:has_region_index(region_type)
  assert(flow_region_tree.is_region(region_type))
  return rawget(self.region_indices, region_type)
end

function region_tree:region_index(region_type)
  assert(flow_region_tree.is_region(region_type))
  assert(rawget(self.region_indices, region_type))
  return self.region_indices[region_type]
end

function region_tree:point_partition(region_type)
  assert(flow_region_tree.is_region(region_type))
  assert(rawget(self.region_point_partitions, region_type))
  return self.region_point_partitions[region_type]
end

function region_tree:is_point(region_type)
  assert(flow_region_tree.is_region(region_type))
  assert(rawget(self.region_is_point, region_type) ~= nil)
  return self.region_is_point[region_type]
end

function region_tree:add_region(region_type, symbol, var_type, annotations, span, is_point)
  assert(flow_region_tree.is_region(region_type) and std.is_symbol(symbol) and
           terralib.types.istype(var_type) and
           annotations and span and
           type(is_point) == "boolean")
  assert(not self:has_region(region_type))
  self.region_var_types[region_type] = var_type
  self.region_annotation_sets[region_type] = annotations
  self.region_spans[region_type] = span
  self.region_is_point[region_type] = is_point

  local value_type = std.as_read(var_type)
  if not is_point and std.is_region(value_type) then
    local partition = std.partition(std.disjoint, std.complete, symbol)
    self.region_point_partitions[region_type] = partition
    std.add_constraint(self, partition, region_type, std.subregion, false)
    self:intern_region_expr(partition, annotations, span)
  end
  if std.is_partition(value_type) or std.is_cross_product(value_type) then
    self:intern_region_expr(value_type:parent_region(), annotations, span)
  end
  if std.is_cross_product(value_type) then
    self:intern_region_expr(value_type:partition(), annotations, span)
  end
end

function region_tree:intern_variable(expr_type, symbol, annotations, span)
  -- Assign a fresh region to non-region symbols.
  local value_type = std.as_read(expr_type)
  local region_type = value_type
  if not flow_region_tree.is_region(value_type) then
    if rawget(self.interned_scalars, symbol) then
      return self.interned_scalars[symbol]
    end

    region_type = std.region(terralib.types.unit)
    for other, _ in self.region_universe:items() do
      std.add_constraint(self, region_type, other, std.disjointness, true)
    end
    self.interned_scalars[symbol] = region_type
  end
  assert(flow_region_tree.is_region(region_type))

  self.region_universe[region_type] = true
  if not self:has_region(region_type) then
    self:add_region(region_type, symbol, expr_type, annotations, span, false)
  end
  return region_type
end

function region_tree:intern_region_expr(expr_type, annotations, span)
  local region_type = std.as_read(expr_type)
  assert(flow_region_tree.is_region(region_type))
  if self:has_region(region_type) then
    return
  end

  self.region_universe[region_type] = true

  local symbol = std.newsymbol(region_type)
  self:add_region(region_type, symbol, expr_type, annotations, span, false)
end

function region_tree:intern_region_point_expr(parent, index, annotations, span)
  assert(std.is_region(parent) and not self:is_point(parent))
  local partition = self:point_partition(parent)
  local subregion
  if index then
    assert(false) -- FIXME: This is currently broken.
    assert(std.issymbol(index.value))
    subregion = partition:subregion_constant(index.value)
  else
    subregion = partition:subregion_dynamic()
  end

  if self:has_region(subregion) then
    return subregion
  end

  self.region_universe[subregion] = true

  local symbol = std.newsymbol(subregion)
  self:add_region(subregion, symbol, subregion, annotations, span, true)
  std.add_constraint(self, subregion, partition, std.subregion, false)
  self:attach_region_index(subregion, index)
  return subregion, symbol
end

function region_tree:attach_region_index(region_type, index)
  assert(flow_region_tree.is_region(region_type))
  assert(self:has_region(region_type))
  self.region_indices[region_type] = index
end

function region_tree:ensure_variable(expr_type, symbol)
  local region_type = std.as_read(expr_type)
  if not flow_region_tree.is_region(region_type) then
    assert(symbol and rawget(self.interned_scalars, symbol))
    region_type = self.interned_scalars[symbol]
  end
  assert(self:has_region(region_type))
  return region_type
end

local function search_constraint_paths(constraints, region_type, path, visited,
                                       predicate)
  assert(not rawget(visited, region_type))
  visited[region_type] = true

  path:insert(region_type)
  if constraints:has(std.subregion) and constraints[std.subregion]:has(region_type) then
    for parent, _ in constraints[std.subregion][region_type]:items() do
      local result = search_constraint_paths(
        constraints, parent, path, visited, predicate)
      if result then
        return result
      end
    end
  else
    if predicate(path) then
      return path
    end
  end
  path:remove()
end

function region_tree:aliased(region_type)
  assert(flow_region_tree.is_region(region_type))
  if std.is_region(region_type) then
    return true
  elseif std.is_partition(region_type) then
    return not region_type:is_disjoint()
  else
    assert(false)
  end
end

function region_tree:can_alias(region_type, other_region_type)
  assert(flow_region_tree.is_region(region_type))
  assert(flow_region_tree.is_region(other_region_type))
  return std.type_maybe_eq(region_type:fspace(), other_region_type:fspace()) and
    not std.check_constraint(
      self, std.constraint(region_type, other_region_type, std.disjointness))
end

function region_tree:ancestors(region_type)
  assert(flow_region_tree.is_region(region_type))
  return search_constraint_paths(
    self.constraints, region_type, terralib.newlist(), {},
    function() return true end)
end

-- Is region_type a subregion of other_region_type?
function region_tree:is_subregion(region_type, other_region_type)
  if not std.type_maybe_eq(region_type:fspace(), other_region_type:fspace()) then
    return false
  end

  return std.check_constraint(
    self, std.constraint(region_type, other_region_type, std.subregion))
end

function region_tree:lowest_common_ancestor(region_type, other_region_type)
  assert(flow_region_tree.is_region(region_type))
  assert(flow_region_tree.is_region(other_region_type))
  return std.search_constraint_predicate(
    self, region_type, {},
    function(cx, ancestor, x)
      if std.check_constraint(
        self, std.constraint(other_region_type, ancestor, std.subregion))
      then
        return ancestor
      end
    end)
end

function region_tree:parent(region_type)
  assert(flow_region_tree.is_region(region_type))
  if self.constraints:has(std.subregion) and
    self.constraints[std.subregion]:has(region_type)
  then
    for parent, _ in self.constraints[std.subregion][region_type]:items() do
      return parent
    end
  end
end

function region_tree:children(region_type)
  assert(flow_region_tree.is_region(region_type))

  local result = terralib.newlist()
  if self.constraints:has(std.subregion) then
    for other, parents in self.constraints[std.subregion]:items() do
      for parent, _ in parents:items() do
        if parent == region_type then
          result:insert(other)
          break
        end
      end
    end
  end
  return result
end

function region_tree:is_sibling(region_type, other_region_type)
  if not std.type_maybe_eq(region_type:fspace(), other_region_type:fspace()) then
    return false
  end

  local is_subregion = std.check_constraint(
    self, std.constraint(region_type, other_region_type, std.subregion))
  local is_superregion = std.check_constraint(
    self, std.constraint(other_region_type, region_type, std.subregion))
  local is_disjoint = std.check_constraint(
    self, std.constraint(region_type, other_region_type, std.disjointness))
  return other_region_type ~= region_type and
    not (is_subregion or is_superregion or is_disjoint)
end

function region_tree:siblings(region_type)
  assert(flow_region_tree.is_region(region_type))

  local siblings = terralib.newlist()
  for other, _ in self.region_universe:items() do
    if self:is_sibling(region_type, other) then
      siblings:insert(other)
    end
  end
  return siblings
end

return flow_region_tree
