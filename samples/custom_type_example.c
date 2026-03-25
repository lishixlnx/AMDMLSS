#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
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
    MLSSstring op = (MLSSstring)opName;  // Cast to match API signature
    
    // Set all MHA parameters from the custom type
    status = mlssSetParameterByEnum(context, op, MLSS_ATTR_MHA_BATCH, (MLSSvoid*)&params->batch_size);
    if (status != MLSS_SUCCESS) return status;
    
    status = mlssSetParameterByEnum(context, op, MLSS_ATTR_MHA_HEADCOUNT, (MLSSvoid*)&params->head_num);
    if (status != MLSS_SUCCESS) return status;
    
    status = mlssSetParameterByEnum(context, op, MLSS_ATTR_MHA_QSEQ, (MLSSvoid*)&params->q_sequence_length);
    if (status != MLSS_SUCCESS) return status;
    
    status = mlssSetParameterByEnum(context, op, MLSS_ATTR_MHA_KVSEQ, (MLSSvoid*)&params->kv_sequence_length);
    if (status != MLSS_SUCCESS) return status;
    
    status = mlssSetParameterByEnum(context, op, MLSS_ATTR_MHA_SIZEHEADS, (MLSSvoid*)&params->head_dim);
    if (status != MLSS_SUCCESS) return status;
    
    status = mlssSetParameterByEnum(context, op, MLSS_ATTR_MHA_SCALE, (MLSSvoid*)&params->scale);
    if (status != MLSS_SUCCESS) return status;
    
    status = mlssSetParameterByEnum(context, op, MLSS_ATTR_MHA_DATATYPE, (MLSSvoid*)&params->data_type);
    if (status != MLSS_SUCCESS) return status;
    
    // Set default values for other required parameters
    MLSSuint32 packing = MLSS_ATTR_CONFIG_MHA_PACKING_UNPACKED;
    status = mlssSetParameterByEnum(context, op, MLSS_ATTR_MHA_PACKING, (MLSSvoid*)&packing);
    if (status != MLSS_SUCCESS) return status;
    
    MLSSuint32 kvDim = 0;
    status = mlssSetParameterByEnum(context, op, MLSS_ATTR_MHA_KDIM, (MLSSvoid*)&kvDim);
    if (status != MLSS_SUCCESS) return status;
    
    status = mlssSetParameterByEnum(context, op, MLSS_ATTR_MHA_VDIM, (MLSSvoid*)&kvDim);
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
    
    status = mlssSetParameterByName(context, op, "qStrides", qStrides);
    if (status != MLSS_SUCCESS) return status;
    
    status = mlssSetParameterByName(context, op, "kStrides", kStrides);
    if (status != MLSS_SUCCESS) return status;
    
    status = mlssSetParameterByName(context, op, "vStrides", vStrides);
    if (status != MLSS_SUCCESS) return status;
    
    status = mlssSetParameterByName(context, op, "outputStrides", outputStrides);
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
    params->batch_size = 2;
    params->head_num = 8;
    params->q_sequence_length = 4096;
    params->kv_sequence_length = 77;
    params->head_dim = 40;
    params->scale = 0.158114f;
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

static int startsWithIgnoreCase(const char* value, const char* prefix)
{
    while (*prefix)
    {
        if (*value == '\0' || tolower((unsigned char)*value) != tolower((unsigned char)*prefix))
        {
            return 0;
        }
        ++value;
        ++prefix;
    }

    return 1;
}

static int formatGfxArgument(const char* arg, char* dest, size_t destSize)
{
    if (arg == NULL || dest == NULL || destSize == 0)
    {
        return 0;
    }

    const char* token = arg;

    if (startsWithIgnoreCase(token, "mlss_gfx"))
    {
        token += 8;
    }
    else if (startsWithIgnoreCase(token, "gfx"))
    {
        token += 3;
    }

    if (*token == '\0')
    {
        return 0;
    }

    char normalized[64] = { 0 };
    size_t idx = 0;

    while (*token != '\0' && idx < (sizeof(normalized) - 1))
    {
        unsigned char ch = (unsigned char)*token;
        if (!isalnum(ch) && ch != '_')
        {
            return 0;
        }

        normalized[idx++] = (char)toupper(ch);
        ++token;
    }

    if (*token != '\0' || idx == 0)
    {
        return 0;
    }

    int written = snprintf(dest, destSize, "MLSS_GFX%s", normalized);
    if (written < 0 || (size_t)written >= destSize)
    {
        return 0;
    }

    return 1;
}

static void printUsage(const char* programName)
{
    printf("Usage: %s [options]\n", programName);
    printf("Options:\n");
    printf("  -g, --gfx <name>       Force a specific GFX target (default auto)\n");
    printf("                         Accepts values like 1100 or gfx1201\n");
    printf("  -h, --help            Show this help message\n");
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

    MLSSstring asic = MLSS_GFXAUTOFIND;
    char customAsic[64] = { 0 };

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-g") == 0 || strcmp(argv[i], "--gfx") == 0)
        {
            if (i + 1 < argc)
            {
                if (!formatGfxArgument(argv[i + 1], customAsic, sizeof(customAsic)))
                {
                    printf("Error: --gfx expects either the gfx number (e.g. 1201) or \"gfx + number\" (eg gfx1201)\n");
                    printUsage(argv[0]);
                    return EXIT_FAILURE;
                }

                asic = customAsic;
                i++;
            }
            else
            {
                printf("Error: --gfx requires a name argument\n");
                printUsage(argv[0]);
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            printUsage(argv[0]);
            return EXIT_SUCCESS;
        }
        else
        {
            printf("Error: Unknown option '%s'\n", argv[i]);
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }
    
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
        for (size_t i = 0; typeInfo.m_supportedOperators[i] != NULL; i++)
        {
            printf("    - %s\n", typeInfo.m_supportedOperators[i]);
        }
    }
    printf("\n");
    
    // Step 4: Create a context and use the custom type
    MLSScontext context = 0;
    MLSSstring opName = MLSS_MHA;
    
    printf("Creating context for %s on %s...\n", opName, asic);
    CHECK_STATUS(mlssCreateContext(&context, asic, opName));

    // Step 5: Create and populate custom type data
    MHAParameters mhaParams = {
        .batch_size = 2,
        .head_num = 8,
        .q_sequence_length = 4096,
        .kv_sequence_length = 77,
        .head_dim = 40,
        .scale = 0.158114f,
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
