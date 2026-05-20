/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */

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
    printf("  -g, --gfx <name>       Force a specific GFX target (default auto)\n");
    printf("                         Accepts values like 1100 or gfx1201\n");
    printf("  -h, --help            Show this help message\n");
    printf("\nExample:\n");
    printf("  %s --verbose 4        Run with DEBUG level verbosity\n", programName);
}

int main(int argc, char* argv[])
{
    MLSSenum verboseLevel = 4;
    //MLSSstring asic = MLSS_GFXAUTOFIND;
    MLSSstring asic = MLSS_GFX1201;
    char customAsic[64] = {0};

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

    printf("=== MVN2 (InstaNorm) Simple Example ===\n");
    printf("Verbose level set to: %d\n\n", verboseLevel);

    /*
     * MVN2 InstaNorm constraints:
     *   - N must be 1
     *   - dataType must be FP16
     *   - crossChannel must be false (per-channel normalization)
     *   - (H * W) must be a multiple of 256
     *   - hasScale and hasBias both true (scale+bias after normalization)
     *   - activation must be IDENTITY (no activation)
     */

    MLSSuint32 n         = 1;
    MLSSuint32 c         = 64;
    MLSSuint32 h         = 256;
    MLSSuint32 w         = 256;
    MLSSfloat32 epsilon  = 1e-5f;
    MLSSbool crossChannel = 0;
    MLSSuint32 sbDims[4] = {1, c, 1, 1};
    MLSSbool hasScale    = 1;
    MLSSbool hasBias     = 1;
    MLSSenum dataType    = MLSS_FLOAT16;
    MLSSenum activation  = MLSS_ACTIVATION_IDENTITY;

    MLSScontext context = 0;
    MLSSbinary* binaries = NULL;
    MLSSsize numBinaries = 0;
    MLSSstring opName = MLSS_MVN;

    MLSSstatus* pStatuses = NULL;
    MLSSsize nStatuses = 0;

    // Step 1) Create context
    CHECK_STATUS(mlssCreateContext(&context, asic, opName));

    // Step 2) Set parameters by enum
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MVN_N, &n));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MVN_C, &c));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MVN_H, &h));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MVN_W, &w));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MVN_EPSILON, &epsilon));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MVN_CROSSCHANNEL, &crossChannel));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MVN_SBDIMS, sbDims));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MVN_HASSCALE, &hasScale));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MVN_HASBIAS, &hasBias));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MVN_DATATYPE, &dataType));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MVN_ACTIVATION, &activation));

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
    CHECK_STATUS(mlssGetBinaries(context, &binaries, &numBinaries));
    CHECK_STATUS(mlssPrintBinaries(binaries, numBinaries));

    printf("\n=== MVN2 (InstaNorm) Simple Example Complete ===\n");

    return EXIT_SUCCESS;
}
