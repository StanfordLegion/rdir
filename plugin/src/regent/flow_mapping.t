-- Copyright (c) 2015-2016, Stanford University. All rights reserved.
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

-- Mappings that statically assign tasks to shards.

-- When the SPMD optimization is used with ForList loops over an index space I
-- as leaves, mappings determine the assignments of indices in I to shards.  A
-- careful choice of mapping may reduce communication between nodes, e.g. in
-- the case of a stencil code, it helps to preserve spacial locality by
-- assigning to each node neighboring partitions of the entire grid.

-- A mapping is a function that takes an index in a structured index space
-- (with one domain) and the size of the index space, and returns a "key"
-- for the index.  The mapping is provided to the `list_ispace` operator,
-- which sorts the indices in the index space by their keys in ascending order.
-- Then, indices-to-shard assignment takes place according to list's ordering
-- (the first `shard_size` indices go to the first shard, etc.)

-- Currently, which mapping to use is controlled by the command line argument
-- `flow-spmd-mapping`, which only takes integers.  The mappings are
-- consequently named by their integer ids starting from 1.

local ast = require("regent/ast")
local std = require("regent/std")

-- WARNING: these mappings only work on 2d structured index spaces.
local mappings = {}
mappings[1] = terra(p : std.int2d, s : std.rect2d) : int -- Column major order.
  return p.__ptr.x + p.__ptr.y * s:size().__ptr.x
end

do -- Hilbert curve.
  local H = 1
  local A = 2
  local B = 3
  local C = 4

  local terra make_int2d(x : int, y : int) : std.int2d
    return std.int2d { __ptr = [std.int2d.impl_type] { x = x, y = y } }
  end

  local terra h(p : std.int2d, size : int, pattern : int) : int
    std.assert(p.__ptr.x < size, "x out of range")
    std.assert(p.__ptr.y < size, "y out of range")

    if size == 1 then
      return 0
    end

    var half = size / 2
    var block = half * half

    if pattern == H then
      if (p.__ptr.x<half) and (p.__ptr.y<half) then
        return h(p, size / 2, A)
      elseif (p.__ptr.x<half) and (p.__ptr.y>=half) then
        return block + h(p - make_int2d(0, half), size / 2, H)
      elseif (p.__ptr.x>=half) and (p.__ptr.y>=half) then
        return block * 2 + h(p - make_int2d(half, half), size / 2, H)
      elseif (p.__ptr.x>=half) and (p.__ptr.y<half) then
        return block * 3 + h(p - make_int2d(half, 0), size / 2, B)
      else
        std.assert(false, "Hilbert curve: impossible.")
      end
    elseif pattern == A then
      if (p.__ptr.x<half) and (p.__ptr.y<half) then
        return h(p, size / 2, H)
      elseif (p.__ptr.x>=half) and (p.__ptr.y<half) then
        return block + h(p - make_int2d(half, 0), size / 2, A)
      elseif (p.__ptr.x>=half) and (p.__ptr.y>=half) then
        return block * 2 + h(p - make_int2d(half, half), size / 2, A)
      elseif (p.__ptr.x<half) and (p.__ptr.y>=half) then
        return block * 3 + h(p - make_int2d(0, half), size / 2, C)
      else
        std.assert(false, "Hilbert curve: impossible.")
      end
    elseif pattern == B then
      if (p.__ptr.x>=half) and (p.__ptr.y>=half) then
        return h(p - make_int2d(half, half), size / 2, C)
      elseif (p.__ptr.x<half) and (p.__ptr.y>=half) then
        return block + h(p - make_int2d(0, half), size / 2, B)
      elseif (p.__ptr.x<half) and (p.__ptr.y<half) then
        return block * 2 + h(p, size / 2, B)
      elseif (p.__ptr.x>=half) and (p.__ptr.y<half) then
        return block * 3 + h(p - make_int2d(half, 0), size / 2, H)
      else
        std.assert(false, "Hilbert curve: impossible.")
      end
    elseif pattern == C then
      if (p.__ptr.x>=half) and (p.__ptr.y>=half) then
        return h(p - make_int2d(half, half), size / 2, B)
      elseif (p.__ptr.x>=half) and (p.__ptr.y<half) then
        return block + h(p - make_int2d(half, 0), size / 2, C)
      elseif (p.__ptr.x<half) and (p.__ptr.y<half) then
        return block * 2 + h(p, size / 2, C)
      elseif (p.__ptr.x<half) and (p.__ptr.y>=half) then
        return block * 3 + h(p - make_int2d(0, half), size / 2, A)
      else
        std.assert(false, "Hilbert curve: impossible.")
      end
    else
      std.assert(false, "Hilbert curve: unknown pattern.")
    end
  end

  mappings[2] = terra(p : std.int2d, s : std.rect2d) : int
    var size = s:size()
    if size.__ptr.x == 2 * size.__ptr.y then
      -- Special case for when x == 2*y and y is a power of 2.
      var half = size.__ptr.y
      if (p - s.lo).__ptr.x < half then
        return h(p - s.lo, half, H)
      else
        return h(p - s.lo - make_int2d(half, 0), half, H) + half * half
      end
    end

    std.assert(size.__ptr.x == size.__ptr.y, "Hilbert curve: plane should be square.")
    std.assert((size.__ptr.x and (size.__ptr.x - 1)) == 0, "Hilbert curve: size should be power of two.")
    return h(p - s.lo, size.__ptr.x, H)
  end
end

mappings[3] = terra(p : std.int2d, s : std.rect2d) : int -- Random shuffle.
  return [std.c.rand]()
end

local spmd_mapping, spmd_mapping_key_type = false, false
local mapping_id = std.config["flow-spmd-mapping"]
if mapping_id > 0 then
  if mappings[mapping_id] then
    local mapping_fn = mappings[mapping_id]
    spmd_mapping = ast.typed.expr.Function {
      value = mapping_fn,
      expr_type = mapping_fn.type,
      annotations = ast.default_annotations(),
      span = ast.trivial_span(),
    }
    spmd_mapping_key_type = mapping_fn.type.returntype
  else
    error("flow_mappings: mapping " .. mapping_id .. " does not exist.")
  end
end

-- Exports the chosen mapping.
return { fn = spmd_mapping, key_type = spmd_mapping_key_type}
