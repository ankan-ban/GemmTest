#include  "Metacommand.h"
#include <dxgi1_6.h>
#include <comdef.h>
#include "d3dx12.h"
#include "utils.h"

#define checkResult(ans) { _checkResult((ans), __FILE__, __LINE__); }
inline void _checkResult(HRESULT hr, const char* file, int line) {
    if (hr != S_OK) {
        _com_error err(hr);
        LPCTSTR errMsg = err.ErrorMessage();

        printf("Error: %s in file %s, line no: %d\n", errMsg, file, line);
        if (abort)
        {
            __debugbreak();
            exit(hr);
        }
    }
}

struct D3D12Alloc 
{
    ID3D12Resource* pResource;
    uint32_t offset;    // offset within pResource (for suballocated resources)
    uint64_t gpuVA;
    D3D12_GPU_DESCRIPTOR_HANDLE descHandle;
};

// handle DX stuff
class D3d12Wrapper
{
private:
    ID3D12Device5 *m_pDevice;
    ID3D12CommandAllocator* m_pCA;
    ID3D12CommandQueue* m_pCQ;
    ID3D12GraphicsCommandList4* m_pCL;
    ID3D12Fence *m_pFence;
    UINT64 m_fenceVal = 0ull;

    ID3D12QueryHeap* m_pQueryHeap;
    D3D12Alloc m_queryResult;

    ID3D12DescriptorHeap *m_pDescHeap;
    static constexpr int MAX_DESCS = 32;
    int nextFreeDescHeapSlot;

public:
    void init(int gpuIndex)
    {
        IDXGIFactory4 *pFactory = nullptr;
        IDXGIAdapter *pAdapter = nullptr;
        checkResult(CreateDXGIFactory2(0, IID_PPV_ARGS(&pFactory)));
        checkResult(pFactory->EnumAdapters(gpuIndex, &pAdapter));
        pFactory->Release();

        checkResult(D3D12CreateDevice(pAdapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_pDevice)));
        pAdapter->Release();

        D3D12_COMMAND_QUEUE_DESC cqDesc = {};
        cqDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

        checkResult(m_pDevice->CreateCommandQueue(&cqDesc, IID_PPV_ARGS(&m_pCQ)));
        checkResult(m_pDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_pCA)));
        checkResult(m_pDevice->CreateCommandList(1, D3D12_COMMAND_LIST_TYPE_DIRECT, m_pCA, nullptr, IID_PPV_ARGS(&m_pCL)));

        D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        heapDesc.NumDescriptors = MAX_DESCS;
        checkResult(m_pDevice->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_pDescHeap)));
        nextFreeDescHeapSlot = 0;

        m_fenceVal = 0ull;
        checkResult(m_pDevice->CreateFence(m_fenceVal, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_pFence)));

        D3D12_QUERY_HEAP_DESC queryHeapDesc = {};
        queryHeapDesc.Count = 2;
        queryHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        queryHeapDesc.NodeMask = 1;
        checkResult(m_pDevice->CreateQueryHeap(&queryHeapDesc, IID_PPV_ARGS(& m_pQueryHeap)));

        createAlloc(sizeof(uint64_t) * 2, D3D12_HEAP_TYPE_READBACK, &m_queryResult);
    }

    void createAlloc(size_t size, D3D12_HEAP_TYPE type, D3D12Alloc* pAlloc) 
    {
        // some alignment
        int factor = ((size - 1)/4) + 1;
        size = factor * 4;

        D3D12_HEAP_PROPERTIES heapDesc = {};
        heapDesc.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heapDesc.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heapDesc.CreationNodeMask = 1;
        heapDesc.VisibleNodeMask = 1;

        if (type == D3D12_HEAP_TYPE_CUSTOM) {
            // Use custom heap type to allow GPU writing to system memory directly
            heapDesc.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
            heapDesc.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
        }

        heapDesc.Type = type;

        D3D12_RESOURCE_DESC bufferDesc = {};
        bufferDesc.MipLevels = 1;
        bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
        bufferDesc.Height = 1;
        if (type == D3D12_HEAP_TYPE_DEFAULT || type == D3D12_HEAP_TYPE_CUSTOM)
            bufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        bufferDesc.DepthOrArraySize = 1;
        bufferDesc.SampleDesc.Count = 1;
        bufferDesc.SampleDesc.Quality = 0;
        bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        D3D12_RESOURCE_STATES resourceState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        if (type == D3D12_HEAP_TYPE_UPLOAD)
            resourceState = D3D12_RESOURCE_STATE_GENERIC_READ;
        else if (type == D3D12_HEAP_TYPE_READBACK)
            resourceState = D3D12_RESOURCE_STATE_COPY_DEST;

        bufferDesc.Width = size;
        checkResult(m_pDevice->CreateCommittedResource(
            &heapDesc, D3D12_HEAP_FLAG_NONE, &bufferDesc, resourceState, nullptr,
            IID_PPV_ARGS(&pAlloc->pResource)));

        pAlloc->offset = 0;
        pAlloc->gpuVA = pAlloc->pResource->GetGPUVirtualAddress();

        // Create desc heap entry for UAV resources.
        if (resourceState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            int slot = nextFreeDescHeapSlot++;

            int handleIncrementSize = m_pDevice->GetDescriptorHandleIncrementSize(
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

            CD3DX12_CPU_DESCRIPTOR_HANDLE cpuDescHandle(m_pDescHeap->GetCPUDescriptorHandleForHeapStart(), slot, handleIncrementSize);

            CD3DX12_GPU_DESCRIPTOR_HANDLE gpuDescHandle(m_pDescHeap->GetGPUDescriptorHandleForHeapStart(), slot, handleIncrementSize);

            D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
            uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
            uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            uavDesc.Buffer.FirstElement = 0;
            uavDesc.Buffer.NumElements = size / 4;

            m_pDevice->CreateUnorderedAccessView(pAlloc->pResource, nullptr, &uavDesc,
                cpuDescHandle);

            pAlloc->descHandle = gpuDescHandle;
        }
    }
    void flushAndWait() 
    {
        m_pCL->Close();
        m_pCQ->ExecuteCommandLists(1, (ID3D12CommandList**)&m_pCL);
        m_pCQ->Signal(m_pFence, ++m_fenceVal);

        // Wait for commands to finish on GPU.
        // (spinloop has lowest latency, we can try event based signal if CPU
        // overhead becomes a bottleneck).
        while (m_pFence->GetCompletedValue() != m_fenceVal) ;

        m_pCA->Reset();
        m_pCL->Reset(m_pCA, NULL);
    }

    ID3D12Device5* getDevice() { return m_pDevice; }
    ID3D12GraphicsCommandList4* getCL() { return m_pCL; }

    void beginTimer()
    {
        m_pCL->EndQuery(m_pQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
    }

    void endTimer()
    {
        m_pCL->EndQuery(m_pQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
        m_pCL->ResolveQueryData(m_pQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, m_queryResult.pResource, 0);
    }

    double getTimeInSeconds()
    {
        uint64_t freq;
        m_pCQ->GetTimestampFrequency(&freq);

        char* pCpuPointer;
        checkResult(m_queryResult.pResource->Map(0, NULL, reinterpret_cast<void**>(&pCpuPointer)));
        uint64_t* TS = (UINT64*)(pCpuPointer);
        double retVal = double((TS[1] - TS[0]) / double(freq));
        m_queryResult.pResource->Unmap(0, nullptr);

        return retVal;
    }

    void uploadData(D3D12Alloc *pAlloc, const void *pData, size_t size)
    {
        // create a staging alloc
        D3D12Alloc staging = {};
        createAlloc(size, D3D12_HEAP_TYPE_UPLOAD, &staging);

        // copy to staging
        char* pCpuPointer;
        checkResult(staging.pResource->Map(0, nullptr, reinterpret_cast<void**>(&pCpuPointer)));
        memcpy(pCpuPointer, pData, size);
        staging.pResource->Unmap(0, nullptr);

        // schedule a copy from staging to the alloc
        m_pCL->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(pAlloc->pResource, 
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST));

        m_pCL->CopyBufferRegion(pAlloc->pResource, pAlloc->offset, staging.pResource, 0, size);

        m_pCL->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(pAlloc->pResource,
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));


        // wait for it to finish
        flushAndWait();

        staging.pResource->Release();
    }
    
    void downloadData(void *pData, D3D12Alloc *pAlloc, size_t size)
    {
        // create a staging alloc
        D3D12Alloc staging = {};
        createAlloc(size, D3D12_HEAP_TYPE_READBACK, &staging);

        // schedule a copy from the alloc to staging
        m_pCL->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(pAlloc->pResource,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE));

        m_pCL->CopyBufferRegion(staging.pResource, 0, pAlloc->pResource, pAlloc->offset, size);

        m_pCL->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(pAlloc->pResource,
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

        // wait for it to finish
        flushAndWait();

        // copy from staging
        char* pCpuPointer;
        checkResult(staging.pResource->Map(0, nullptr, reinterpret_cast<void**>(&pCpuPointer)));
        memcpy(pData, pCpuPointer, size);
        staging.pResource->Unmap(0, nullptr);

        staging.pResource->Release();
    }

    void destroyAlloc(D3D12Alloc *pAlloc)
    {
        pAlloc->pResource->Release();
    }

    void destroy()
    {
        m_pFence->Release();
        m_pDescHeap->Release();
        m_queryResult.pResource->Release();
        m_pQueryHeap->Release();
        m_pCA->Release();
        m_pCQ->Release();
        m_pCL->Release();
        m_pDevice->Release();
    }
};

D3d12Wrapper g_DXWrapper;

// get descriptor for row-major matrix (or batch of 'n' matrices)
static void getTensorDesc(TensorDesc* outDesc, int n, int rows, int cols, bool fp16 = true) 
{
    outDesc->DimensionCount = 4;
    outDesc->DataType = fp16 ? 1 : 0;

    outDesc->Size[0] = n;
    outDesc->Size[1] = 1;
    outDesc->Size[2] = rows;    // height
    outDesc->Size[3] = cols;    // width

    outDesc->Stride[3] = 1;
    outDesc->Stride[2] = cols;
    outDesc->Stride[1] = rows * cols;
    outDesc->Stride[0] = rows * cols;

    for (int i = 0; i < 4; i++) outDesc->StrideAlignment[i] = 1;
    outDesc->BaseAlignmentInBytes = 4096;
    outDesc->PhysicalSizeInElements = n * rows * cols;
}


int main()
{
    const int gpuToUse = 0;
    g_DXWrapper.init(gpuToUse);

    const int useFp16 = true;
    const int M = 1024;
    const int N = 256;
    const int K = 256;
    const int batch = 36;
    const int elementSize = useFp16 ? sizeof(uint16_t) : sizeof(float);

    void *cpuA = malloc(M*K*batch*elementSize);
    void *cpuB = malloc(K*N*batch*elementSize);
    void *cpuOut = malloc(M*N*batch*elementSize);
    void *cpuRef = malloc(M*N*batch*elementSize);

    fillRandomArray(cpuA, M*K*batch, useFp16);
    fillRandomArray(cpuB, K*N*batch, useFp16);

    GemmCreateDesc createDesc = {};
    getTensorDesc(&createDesc.DescOut, batch, M, N, useFp16);
    getTensorDesc(&createDesc.DescA, batch, M, K, useFp16);
    getTensorDesc(&createDesc.DescB, batch, K, N, useFp16);
    createDesc.cMatrixNull = 1;
    createDesc.ActivationIsNull = 1;
    createDesc.Alpha = 1.0;
    createDesc.Beta = 0.0;
    createDesc.Precision = useFp16 ? 1 : 0; // 0 - fp32, 1 - fp16

    ID3D12MetaCommand *pMetacommand = nullptr;
    checkResult(g_DXWrapper.getDevice()->CreateMetaCommand(GemmGuid, 1, &createDesc, sizeof(createDesc), IID_PPV_ARGS(&pMetacommand)));

    size_t ASize = pMetacommand->GetRequiredParameterResourceSize(D3D12_META_COMMAND_PARAMETER_STAGE_EXECUTION, 0);
    size_t BSize = pMetacommand->GetRequiredParameterResourceSize(D3D12_META_COMMAND_PARAMETER_STAGE_EXECUTION, 1);
    size_t OutSize = pMetacommand->GetRequiredParameterResourceSize(D3D12_META_COMMAND_PARAMETER_STAGE_EXECUTION, 3);

    size_t persistentSize = pMetacommand->GetRequiredParameterResourceSize(D3D12_META_COMMAND_PARAMETER_STAGE_EXECUTION, 4);
    size_t tempSize = pMetacommand->GetRequiredParameterResourceSize(D3D12_META_COMMAND_PARAMETER_STAGE_EXECUTION, 5);
    printf("\nPersistent size: %llu, temp size: %llu\n", persistentSize, tempSize);

    D3D12Alloc persistent = {}, temperory = {}, A = {}, B = {}, Out = {};
    g_DXWrapper.createAlloc(ASize, D3D12_HEAP_TYPE_DEFAULT, &A);
    g_DXWrapper.createAlloc(BSize, D3D12_HEAP_TYPE_DEFAULT, &B);
    g_DXWrapper.createAlloc(OutSize, D3D12_HEAP_TYPE_DEFAULT, &Out);

    g_DXWrapper.uploadData(&A, cpuA, ASize);
    g_DXWrapper.uploadData(&B, cpuB, BSize);

    if (persistentSize)
        g_DXWrapper.createAlloc(persistentSize, D3D12_HEAP_TYPE_DEFAULT, &persistent);  // huge alloc - driver bug!
    if (tempSize)
        g_DXWrapper.createAlloc(tempSize, D3D12_HEAP_TYPE_DEFAULT, &persistent);

    GemmInitDesc initDesc = {};
    initDesc.PersistentResource = persistent.descHandle;

    g_DXWrapper.getCL()->InitializeMetaCommand(pMetacommand, &initDesc, sizeof(initDesc));

    GemmExecuteDesc execDesc = {};
    execDesc.AResource = A.descHandle;
    execDesc.BResource = B.descHandle;
    execDesc.OutputResource = Out.descHandle;
    execDesc.PersistentResource = persistent.descHandle;
    execDesc.TemporaryResource = temperory.descHandle;

    int loops = 10;
    int iterPerLoop = 100;

    for (int i = 0; i < loops; i++)
    {
        g_DXWrapper.beginTimer();

        for (int j=0;j< iterPerLoop;j++)
            g_DXWrapper.getCL()->ExecuteMetaCommand(pMetacommand, &execDesc, sizeof(execDesc));

        g_DXWrapper.endTimer();

        g_DXWrapper.flushAndWait();
        double time = g_DXWrapper.getTimeInSeconds();
        double flops = (double(M) * N * K * 2 * iterPerLoop * batch) / time;
        double bps = iterPerLoop * batch * (double(M)*N + M * K + K * N) * elementSize / time;
        printf("\nTime taken: %g ms, TFlops: %g, GBps: %g\n", time * 1000 / iterPerLoop, flops / 1000000000000.0, bps / 1000000000.0);
    }

    g_DXWrapper.downloadData(cpuOut, &Out, OutSize);
    matrixMulCPU(M, N, K, batch, cpuRef, cpuA, cpuB, useFp16);
    compareResults(cpuOut, cpuRef, batch*M*N, useFp16);

    if(persistentSize)
        g_DXWrapper.destroyAlloc(&persistent);

    if (tempSize)
        g_DXWrapper.destroyAlloc(&temperory);

    g_DXWrapper.destroyAlloc(&A);
    g_DXWrapper.destroyAlloc(&B);
    g_DXWrapper.destroyAlloc(&Out);
    g_DXWrapper.destroy();

    free(cpuA);
    free(cpuB);
    free(cpuOut);
    free(cpuRef);

    getchar();
}