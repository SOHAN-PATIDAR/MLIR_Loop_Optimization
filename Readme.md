# Affine Loop Interchange Pass

## Overview

This code snippet is part of the `AffineLoopInterchangePass` in the MLIR (Multi-Level Intermediate Representation) framework, specifically within the `getAccessMatrix` function. The purpose of this code is to construct an access matrix for a memory operation (load or store) in an affine loop nest. The access matrix represents how loop indices contribute to the memory access expressions, which is crucial for analyzing loop transformations like loop interchange to optimize locality and parallelism.

## What the Code Does

The code iterates over the results of an affine map associated with a memory operation (e.g., `AffineLoadOp` or `AffineStoreOp`) and builds a matrix `mat` of size `[rank x depth]`, where:

- `rank` is the number of subscript expressions in the affine map (i.e., the number of dimensions in the memory access).
- `depth` is the nesting depth of the loop containing the memory operation (i.e., the number of loop indices).

Each element `mat[dim][d]` represents the coefficient of the loop index at depth `d` in the affine expression for dimension `dim`. This matrix is used to analyze temporal and spatial reuse in the loop nest.

## How It Works

1. **Outer Loop**: Iterates over each dimension `dim` of the affine map (up to `rank`).
   - Retrieves the affine expression (`expr`) for the current dimension from the map's results.
2. **Inner Loop**: Iterates over each loop depth `d` (up to `depth`).
   - Initializes a coefficient `coeff` to 0 for the current dimension and depth.
3. **Coefficient Extraction**:
   - If the expression is a binary operation (`AffineBinaryOpExpr`):
     - Checks if it's a multiplication (`Mul`).
     - If either the left-hand side (LHS) or right-hand side (RHS) is a constant (`AffineConstantExpr`), extracts its value as the coefficient.
     - If the expression is not a multiplication but depends on the loop index at depth `d` (via `isFunctionOfDim`), sets `coeff` to 1.
   - If the expression is not a binary operation but depends on the loop index at depth `d`, sets `coeff` to 1.
4. **Matrix Update**: Stores the computed coefficient in `mat[dim][d]`.
5. **Return**: Returns the completed access matrix `mat`.

## Setup and How to Use

1. Clone llvm project from git and checkout to version 19.1.7
    * git clone https://github.com/llvm/llvm-project.git
    * cd llvm-project
    * git checkout b9d27ac252265839354fffeacaa8f39377ed7424

2. Configure and build mlir and run LLVM tests:
    * cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DCMAKE_BUILD_TYPE=Debug \
   -DLLVM_ENABLE_ASSERTIONS=ON

    * cmake --build . --target check-mlir

3. Write your code files and build it using
    * ninja // make sure you're in build directory

4. testing the code file uisng Filecheck
    *  build/bin/mlir-opt -affine-loop-interchange path_to_test_file | build/bin/FileCheck path_to_test_file

* **NOTE**: Modify below files to make sure that your pass is getting detected and build
    * llvm-project/mlir/include/mlir/Dialect/Affine/Passes.td
    * llvm-project/mlir/include/mlir/Dialect/Affine/Passes.h
    * llvm-project/mlir/lib/Dialect/Affine/Transforms/CMakeLists.txt

## Failing Test Cases
For the loop interchange pass that I implemneted 3 test cases are failing 4, 7 and 9
