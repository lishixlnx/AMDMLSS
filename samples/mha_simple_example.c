#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>

#include <amdmlss/amdmlss_api.h>

void checkStatus(MLSSstatus status, int line)
{
    if (status != MLSS_SUCCESS)
    {
        MLSSstring err = mlssGetErrorString(status);

        printf("Failed at line: %d, :%s\n", line, err);
        free(err);
        exit(EXIT_FAILURE);
    }
}

#define CHECK_STATUS(status) checkStatus((status), __LINE__)

void printUsage(const char* programName)
{
    printf("Usage: %s [options]\n", programName);
    printf("Options:\n");
    printf("  -v, --verbose <level>  Set verbose level (0-5)\n");
    printf("                         0 = NONE (no output)\n");
    printf("                         1 = ERROR\n");
    printf("                         2 = WARNING\n");
    printf("                         3 = INFO\n");
    printf("                         4 = DEBUG (default)\n");
    printf("                         5 = TRACE\n");
    printf("  -h, --help            Show this help message\n");
    printf("\nExample:\n");
    printf("  %s --verbose 4        Run with DEBUG level verbosity\n", programName);
}

int main(int argc, char* argv[])
{
    // Default to DEBUG level verbose mode
    MLSSenum verboseLevel = 4; // DEBUG level
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0)
        {
            if (i + 1 < argc)
            {
                verboseLevel = atoi(argv[i + 1]);
                if (verboseLevel > 5)
                {
                    printf("Warning: Invalid verbose level %d, using INFO (3) instead\n", verboseLevel);
                    verboseLevel = 3;
                }
                i++; // Skip the next argument since we consumed it
            }
            else
            {
                printf("Error: --verbose requires a level argument\n");
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
    
    // Set the verbose level
    CHECK_STATUS(mlssSetVerboseLevel(verboseLevel));
    
    printf("=== MHA Simple Example ===\n");
    printf("Verbose level set to: %d\n\n", verboseLevel);

    MLSSuint32 batch_size = 2;
    MLSSuint32 head_num = 8;
    MLSSuint32 q_sequence_length = 4096;
    MLSSuint32 kv_sequence_length = 77;
    MLSSuint32 head_dim = 40;
    MLSSuint32 packing = MLSS_ATTR_CONFIG_MHA_PACKING_UNPACKED;
    MLSSfloat32 scale = 0.158114f;
    MLSSenum data_type = MLSS_FLOAT16;
    MLSSuint32 kvDim = 0;


    MLSScontext context = 0;
    MLSSbinary* binaries = NULL;
    MLSSsize n = 0;
    MLSSuint32** constants = NULL;
    MLSSsize* numConstants = NULL;

    MLSSstring asic = MLSS_GFXAUTOFIND;
    MLSSstring opName = MLSS_MHA;

    MLSSstatus* pStatuses = NULL;
    MLSSsize nStatuses = 0;

    // Step 1) Create context
    CHECK_STATUS(mlssCreateContext(&context, asic, opName));

    // Step 2) Set parameters
    
    // Set the parameters by name...
    //CHECK_STATUS(mlssSetParameterByName(&context, opName, "batchSize", &batch_size));
    //CHECK_STATUS(mlssSetParameterByName(&context, opName, "qSeqLength", &q_sequence_length));
    //CHECK_STATUS(mlssSetParameterByName(&context, opName, "kvSeqLength", &kv_sequence_length));
    //CHECK_STATUS(mlssSetParameterByName(&context, opName, "kDim", &kvDim));
    //CHECK_STATUS(mlssSetParameterByName(&context, opName, "vDim", &kvDim));
    //CHECK_STATUS(mlssSetParameterByName(&context, opName, "sizeHeads", &head_dim));
    //CHECK_STATUS(mlssSetParameterByName(&context, opName, "packing", &packing));
    //CHECK_STATUS(mlssSetParameterByName(&context, opName, "headCount", &head_num));
    //CHECK_STATUS(mlssSetParameterByName(&context, opName, "scale", &scale));
    //CHECK_STATUS(mlssSetParameterByName(&context, opName, "dataType", &data_type));

    // ... or by enum

    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MHA_BATCH, &batch_size));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MHA_QSEQ, &q_sequence_length));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MHA_KVSEQ, &kv_sequence_length));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MHA_KDIM, &kvDim));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MHA_VDIM, &kvDim));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MHA_SIZEHEADS, &head_dim));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MHA_PACKING, &packing));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MHA_HEADCOUNT, &head_num));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MHA_SCALE, &scale));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MHA_DATATYPE, &data_type));

    MLSSuint32 qStrides[4] = { q_sequence_length * head_num * head_dim, head_dim, head_num * head_dim, 1 };
    MLSSuint32 kStrides[4] = { kv_sequence_length * head_num * head_dim, head_dim, head_num * head_dim, 1 };
    MLSSuint32 vStrides[4] = { kv_sequence_length * head_num * head_dim, head_dim, 1, head_num * head_dim };
    MLSSuint32 outputStrides[4] = { q_sequence_length * head_num * head_dim, head_dim, head_num * head_dim, 1 };

    CHECK_STATUS(mlssSetParameterByName(&context, opName, "qStrides", qStrides));
    CHECK_STATUS(mlssSetParameterByName(&context, opName, "kStrides", kStrides));
    CHECK_STATUS(mlssSetParameterByName(&context, opName, "vStrides", vStrides));
    CHECK_STATUS(mlssSetParameterByName(&context, opName, "outputStrides", outputStrides));

    CHECK_STATUS(mlssPrintParameters(context, opName));

    if (mlssGetCaps(context, &pStatuses, &nStatuses) != MLSS_SUCCESS)
    {
        printf("Failed to get caps\n");
        return EXIT_FAILURE;
    }
    else
    {
        printf("Got caps\n");
    }

    // Step 3) Get binaries.

    CHECK_STATUS(mlssGetBinaries(context, &binaries, &n));

    CHECK_STATUS(mlssPrintBinaries(binaries, n));

    printf("\n=== MHA Simple Example Complete ===\n");

    return EXIT_SUCCESS;
}
