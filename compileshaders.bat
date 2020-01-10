del shaders.h

dxc /Tcs_6_2 /EMatrixMul /Vn g_MatrixMul_Fp32 /Fh temp.txt Shaders.hlsl
type temp.txt >> shaders.h
del temp.txt

REM fp16 math runs much slower on AMD Vega7! compiler perf issue! ??
REM On Nvidia Turing it's much faster
REM dxc /Tcs_6_2 /EMatrixMul /Vn g_MatrixMul_Fp16 /DFP16_IO=1 /DUSE_FP16_MATH=1 /Fh temp.txt Shaders.hlsl -enable-16bit-types
dxc /Tcs_6_2 /EMatrixMul /Vn g_MatrixMul_Fp16 /DFP16_IO=1 /Fh temp.txt Shaders.hlsl -enable-16bit-types
type temp.txt >> shaders.h
del temp.txt

pause