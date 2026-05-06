/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

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

    char normalized[64] = {0};
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
    printf("  -g, --gfx <name>       Force a specific GFX target (default gfx1201)\n");
    printf("                         Accepts values like 1100 or gfx1201\n");
    printf("  -h, --help            Show this help message\n");
    printf("\nExample:\n");
    printf("  %s --verbose 4        Run with DEBUG level verbosity\n", programName);
}

int main(int argc, char* argv[])
{
    MLSSenum  verboseLevel = 4;
    MLSSstring asic        = MLSS_GFX1201;
    char       customAsic[64] = {0};

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
                i++;
            }
            else
            {
                printf("Error: --verbose requires a level argument\n");
                printUsage(argv[0]);
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "-g") == 0 || strcmp(argv[i], "--gfx") == 0)
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

    CHECK_STATUS(mlssSetVerboseLevel(verboseLevel));

    printf("=== GEMM HIP MxN Simple Example ===\n");
    printf("Verbose level set to: %d\n\n", verboseLevel);

    /*
     * GEMM HIP MxN constraints (mirrors dxcp::HipGemm):
     *   - GFX11x or GFX12x architecture
     *   - dataType = FLOAT16 (precision must be FLOAT16 or default)
     *   - transA = false (column-major A is unsupported)
     *   - M, N, K >= 16; N and K must be even
     *   - alpha = 1.0 (fused into the kernel)
     *   - activation must not be SOFTMAX/LOG_SOFTMAX/HARDMAX
     *
     * The default 512x512x512 NN FP16 problem chosen below is selected by
     * the decision tree to a WMMA tile shader (e.g. 64x128x32 or 128x128x16
     * on GFX12).
     */

    MLSSuint32  m            = 512u;
    MLSSuint32  n            = 512u;
    MLSSuint32  k            = 512u;
    MLSSuint32  batch        = 1u;
    MLSSfloat32 alpha        = 1.0f;
    MLSSfloat32 beta         = 0.0f;
    MLSSbool    hasC         = 0;
    MLSSbool    transA       = 0;
    MLSSbool    transB       = 0;
    MLSSenum    dataType     = MLSS_FLOAT16;
    MLSSenum    precision    = MLSS_PRECISION_FLOAT16;
    MLSSenum    activation   = MLSS_ACTIVATION_IDENTITY;

    MLSScontext context     = 0;
    MLSSbinary* binaries    = NULL;
    MLSSsize    numBinaries = 0;
    MLSSstring  opName      = MLSS_GEMM;

    MLSSstatus* pStatuses = NULL;
    MLSSsize    nStatuses = 0;

    // Step 1) Create context
    CHECK_STATUS(mlssCreateContext(&context, asic, opName));

    // Step 2) Set parameters by enum
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_GEMM_M,          &m));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_GEMM_N,          &n));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_GEMM_K,          &k));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_GEMM_BATCH,      &batch));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_GEMM_ALPHA,      &alpha));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_GEMM_BETA,       &beta));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_GEMM_HASC,       &hasC));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_GEMM_TRANSA,     &transA));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_GEMM_TRANSB,     &transB));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_GEMM_DATATYPE,   &dataType));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_GEMM_PRECISION,  &precision));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_GEMM_ACTIVATION, &activation));

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

    // Step 3) Get binaries (HIP backend should win the dispatch).
    CHECK_STATUS(mlssGetBinaries(context, &binaries, &numBinaries));
    CHECK_STATUS(mlssPrintBinaries(binaries, numBinaries));

    printf("\n=== GEMM HIP MxN Simple Example Complete ===\n");

    return EXIT_SUCCESS;
}
