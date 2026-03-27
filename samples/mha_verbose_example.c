#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include <amdmlss/amdmlss_api.h>

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

int main()
{
    printf("=== MHA Verbose Mode Example ===\n\n");

    // Enable verbose mode at INFO level
    printf("Enabling verbose mode at INFO level...\n");
    CHECK_STATUS(mlssSetVerboseLevel(3)); // INFO level

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

    MLSSstring asic = MLSS_GFX1201;
    MLSSstring opName = MLSS_MHA;

    MLSSstatus* pStatuses = NULL;
    MLSSsize nStatuses = 0;

    // Step 1) Create context
    printf("\n--- Creating context ---\n");
    CHECK_STATUS(mlssCreateContext(&context, asic, opName));

    // Step 2) Set parameters
    printf("\n--- Setting parameters ---\n");

    // Set parameters by enum
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

    MLSSuint32 qStrides[4] = {q_sequence_length * head_num * head_dim, head_dim, head_num * head_dim, 1};
    MLSSuint32 kStrides[4] = {kv_sequence_length * head_num * head_dim, head_dim, head_num * head_dim, 1};
    MLSSuint32 vStrides[4] = {kv_sequence_length * head_num * head_dim, head_dim, 1, head_num * head_dim};
    MLSSuint32 outputStrides[4] = {q_sequence_length * head_num * head_dim, head_dim, head_num * head_dim, 1};

    CHECK_STATUS(mlssSetParameterByName(&context, opName, "qStrides", qStrides));
    CHECK_STATUS(mlssSetParameterByName(&context, opName, "kStrides", kStrides));
    CHECK_STATUS(mlssSetParameterByName(&context, opName, "vStrides", vStrides));
    CHECK_STATUS(mlssSetParameterByName(&context, opName, "outputStrides", outputStrides));

    // Print parameters with verbose mode enabled
    printf("\n--- Printing parameters (verbose mode enabled) ---\n");
    CHECK_STATUS(mlssPrintParameters(context, opName));

    // Change verbose level to DEBUG
    printf("\n--- Changing verbose level to DEBUG ---\n");
    CHECK_STATUS(mlssSetVerboseLevel(4)); // DEBUG level

    // Get capabilities
    printf("\n--- Getting capabilities ---\n");
    if (mlssGetCaps(context, &pStatuses, &nStatuses) != MLSS_SUCCESS)
    {
        printf("Failed to get caps\n");
        return EXIT_FAILURE;
    }
    else
    {
        printf("Got caps successfully\n");
    }

    // Step 3) Get binaries
    printf("\n--- Getting binaries ---\n");
    CHECK_STATUS(mlssGetBinaries(context, &binaries, &n));

    // Print binaries with verbose mode
    printf("\n--- Printing binaries (verbose mode enabled) ---\n");
    CHECK_STATUS(mlssPrintBinaries(binaries, n));

    // Demonstrate disabling verbose mode
    printf("\n--- Disabling verbose mode ---\n");
    CHECK_STATUS(mlssDisableVerboseMode());

    // Try printing again with verbose mode disabled
    printf("\n--- Printing parameters (verbose mode disabled) ---\n");
    CHECK_STATUS(mlssPrintParameters(context, opName));

    // Re-enable verbose mode at WARNING level
    printf("\n--- Re-enabling verbose mode at WARNING level ---\n");
    CHECK_STATUS(mlssSetVerboseLevel(2)); // WARNING level

    // Check current verbose level
    MLSSenum currentLevel = mlssGetVerboseLevel();
    printf("Current verbose level: %d\n", currentLevel);

    printf("\n=== MHA Verbose Mode Example Complete ===\n");

    return EXIT_SUCCESS;
}
