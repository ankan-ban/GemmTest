del shaders.h

REM dxc /Tcs_6_2 /EInputTransform_FP16 /DFP16_IO=1 /DUSE_FP16_MATH=1 /Fh temp.txt Shaders.hlsl  -enable-16bit-types

dxc /Tcs_6_2 /EMatrixMul /Vn g_MatrixMul_Fp32 /Fh temp.txt Shaders.hlsl
type temp.txt >> shaders.h
del temp.txt

dxc /Tcs_6_2 /EMatrixMul /Vn g_MatrixMul_Fp16 /DFP16_IO=1 /DUSE_FP16_MATH=1 /Fh temp.txt Shaders.hlsl -enable-16bit-types
type temp.txt >> shaders.h
del temp.txt

pause