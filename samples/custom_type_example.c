#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "amdmlss/amdmlss_api.h"

// Example custom type for MHA parameters
typedef struct {
    MLSSuint32 batch_size;
    MLSSuint32 head_num;
    MLSSuint32 q_sequence_length;
    MLSSuint32 kv_sequence_length;
    MLSSuint32 head_dim;
    MLSSfloat32 scale;
    MLSSenum data_type;
} MHAParameters;

// Custom type setter function
MLSSstatus setMHAParameters(MLSScontext* context, const MLSSchar* opName, const MLSSvoid* data)
{
    const MHAParameters* params = (const MHAParameters*)data;
    MLSSstatus status;
    
    // Set all MHA parameters from the custom type
    status = mlssSetParameterByEnum(context, opName, MLSS_ATTR_MHA_BATCH, (MLSSvoid*)&params->batch_size);
    if (status != MLSS_SUCCESS) return status;
    
    status = mlssSetParameterByEnum(context, opName, MLSS_ATTR_MHA_HEADCOUNT, (MLSSvoid*)&params->head_num);
    if (status != MLSS_SUCCESS) return status;
    
    status = mlssSetParameterByEnum(context, opName, MLSS_ATTR_MHA_QSEQ, (MLSSvoid*)&params->q_sequence_length);
    if (status != MLSS_SUCCESS) return status;
    
    status = mlssSetParameterByEnum(context, opName, MLSS_ATTR_MHA_KVSEQ, (MLSSvoid*)&params->kv_sequence_length);
    if (status != MLSS_SUCCESS) return status;
    
    status = mlssSetParameterByEnum(context, opName, MLSS_ATTR_MHA_SIZEHEADS, (MLSSvoid*)&params->head_dim);
    if (status != MLSS_SUCCESS) return status;
    
    status = mlssSetParameterByEnum(context, opName, MLSS_ATTR_MHA_SCALE, (MLSSvoid*)&params->scale);
    if (status != MLSS_SUCCESS) return status;
    
    status = mlssSetParameterByEnum(context, opName, MLSS_ATTR_MHA_DATATYPE, (MLSSvoid*)&params->data_type);
    if (status != MLSS_SUCCESS) return status;
    
    // Set default values for other required parameters
    MLSSuint32 packing = MLSS_ATTR_CONFIG_MHA_PACKING_UNPACKED;
    status = mlssSetParameterByEnum(context, opName, MLSS_ATTR_MHA_PACKING, (MLSSvoid*)&packing);
    if (status != MLSS_SUCCESS) return status;
    
    MLSSuint32 kvDim = 0;
    status = mlssSetParameterByEnum(context, opName, MLSS_ATTR_MHA_KDIM, (MLSSvoid*)&kvDim);
    if (status != MLSS_SUCCESS) return status;
    
    status = mlssSetParameterByEnum(context, opName, MLSS_ATTR_MHA_VDIM, (MLSSvoid*)&kvDim);
    if (status != MLSS_SUCCESS) return status;
    
    // Set strides
    MLSSuint32 qStrides[4] = {
        params->q_sequence_length * params->head_num * params->head_dim,
        params->head_dim,
        params->head_num * params->head_dim,
        1
    };
    MLSSuint32 kStrides[4] = {
        params->kv_sequence_length * params->head_num * params->head_dim,
        params->head_dim,
        params->head_num * params->head_dim,
        1
    };
    MLSSuint32 vStrides[4] = {
        params->kv_sequence_length * params->head_num * params->head_dim,
        params->head_dim,
        1,
        params->head_num * params->head_dim
    };
    MLSSuint32 outputStrides[4] = {
        params->q_sequence_length * params->head_num * params->head_dim,
        params->head_dim,
        params->head_num * params->head_dim,
        1
    };
    
    status = mlssSetParameterByName(context, opName, "qStrides", qStrides);
    if (status != MLSS_SUCCESS) return status;
    
    status = mlssSetParameterByName(context, opName, "kStrides", kStrides);
    if (status != MLSS_SUCCESS) return status;
    
    status = mlssSetParameterByName(context, opName, "vStrides", vStrides);
    if (status != MLSS_SUCCESS) return status;
    
    status = mlssSetParameterByName(context, opName, "outputStrides", outputStrides);
    if (status != MLSS_SUCCESS) return status;
    
    return MLSS_SUCCESS;
}

// Custom type getter function (retrieves parameters from context)
MLSSstatus getMHAParameters(const MLSScontext context, const MLSSchar* opName, MLSSvoid** data)
{
    // For this example, we'll just return a dummy implementation
    // In a real implementation, you would query the context for the current values
    MHAParameters* params = (MHAParameters*)malloc(sizeof(MHAParameters));
    if (!params) return MLSS_ERROR_BAD_MEMORY_ALLOCATION;
    
    // Initialize with default values
    params->batch_size = 1;
    params->head_num = 4;
    params->q_sequence_length = 256;
    params->kv_sequence_length = 77;
    params->head_dim = 32;
    params->scale = 0.177f;
    params->data_type = MLSS_FLOAT16;
    
    *data = params;
    return MLSS_SUCCESS;
}

// Custom type print function
void printMHAParameters(const MLSSvoid* data)
{
    const MHAParameters* params = (const MHAParameters*)data;
    printf("MHA Parameters:\n");
    printf("  Batch Size: %u\n", params->batch_size);
    printf("  Head Count: %u\n", params->head_num);
    printf("  Q Sequence Length: %u\n", params->q_sequence_length);
    printf("  KV Sequence Length: %u\n", params->kv_sequence_length);
    printf("  Head Dimension: %u\n", params->head_dim);
    printf("  Scale: %f\n", params->scale);
    printf("  Data Type: %u\n", params->data_type);
}

void checkStatus(MLSSstatus status, int line)
{
    if (status != MLSS_SUCCESS)
    {
        MLSSstring err = mlssGetErrorString(status);
        printf("Failed at line: %d, :%s\n", line, err);
        exit(EXIT_FAILURE);
    }
}

#define CHECK_STATUS(status) checkStatus((status), __LINE__)

int main(int argc, char* argv[])
{
    printf("=== Custom Type Registration Example ===\n\n");
    
    // Step 1: Register the custom type
    printf("Registering custom MHA parameter type...\n");
    MLSSenum mhaTypeId = mlssRegisterCustomType(
        "MHAParameters",
        setMHAParameters,
        getMHAParameters,
        MLSS_MHA,           // This custom type can be used with MHA operator
        MLSS_GQA,           // Also supports GQA operator
        MLSS_END_LIST       // Terminate the list
    );
    
    if (mhaTypeId == 0)
    {
        printf("Failed to register custom type!\n");
        return EXIT_FAILURE;
    }
    
    printf("Custom type registered with ID: %u\n\n", mhaTypeId);
    
    // Step 2: Check if the type is registered
    if (mlssIsCustomType(mhaTypeId))
    {
        printf("Type ID %u is confirmed as a custom type\n", mhaTypeId);
    }
    
    // Step 3: Get information about the custom type
    MLSScustomtypeinfo typeInfo;
    CHECK_STATUS(mlssGetCustomTypeInfo(mhaTypeId, &typeInfo));
    printf("Custom type info:\n");
    printf("  Name: %s\n", typeInfo.m_typeName);
    printf("  Type ID: %u\n", typeInfo.m_typeId);
    printf("  Supported operators:\n");
    if (typeInfo.m_supportedOperators)
    {
        for (int i = 0; typeInfo.m_supportedOperators[i] != NULL; i++)
        {
            printf("    - %s\n", typeInfo.m_supportedOperators[i]);
        }
    }
    printf("\n");
    
    // Step 4: Create a context and use the custom type
    MLSScontext context = 0;
    MLSSstring asic = MLSS_GFX1201;
    MLSSstring opName = MLSS_MHA;
    
    printf("Creating context for %s on %s...\n", opName, asic);
    CHECK_STATUS(mlssCreateContext(&context, asic, opName));
    
    // Step 5: Create and populate custom type data
    MHAParameters mhaParams = {
        .batch_size = 1,
        .head_num = 4,
        .q_sequence_length = 256,
        .kv_sequence_length = 77,
        .head_dim = 32,
        .scale = 0.177f,
        .data_type = MLSS_FLOAT16
    };
    
    printf("\nSetting parameters using custom type...\n");
    printMHAParameters(&mhaParams);
    
    // Step 6: Use the custom type with mlssSetParameterByNameTyped
    CHECK_STATUS(mlssSetParameterByNameTyped(
        &context,
        opName,
        "mhaParams",  // Parameter name (not used for custom types)
        mhaTypeId,    // Use our custom type ID
        &mhaParams    // Pass the custom type data
    ));
    
    printf("\nParameters set successfully using custom type!\n");
    
    // Step 7: Print the parameters to verify they were set
    printf("\nVerifying parameters were set correctly:\n");
    CHECK_STATUS(mlssPrintParameters(context, opName));
    
    // Step 8: Test getting binaries (this validates the parameters)
    MLSSbinary* binaries = NULL;
    MLSSsize numBinaries = 0;
    MLSSstatus* pStatuses = NULL;
    MLSSsize nStatuses = 0;
    
    printf("\nGetting capabilities...\n");
    if (mlssGetCaps(context, &pStatuses, &nStatuses) == MLSS_SUCCESS)
    {
        printf("Capabilities check passed\n");
        
        printf("\nGetting binaries...\n");
        CHECK_STATUS(mlssGetBinaries(context, &binaries, &numBinaries));
        printf("Successfully got %zu binaries\n", numBinaries);
    }
    else
    {
        printf("Capabilities check failed (this is expected if GPU is not available)\n");
    }
    
    // Step 9: Unregister the custom type
    printf("\nUnregistering custom type...\n");
    CHECK_STATUS(mlssUnregisterCustomType(mhaTypeId));
    
    // Verify it's no longer registered
    if (!mlssIsCustomType(mhaTypeId))
    {
        printf("Custom type successfully unregistered\n");
    }
    
    printf("\n=== Custom Type Example Complete ===\n");
    
    return EXIT_SUCCESS;
}
