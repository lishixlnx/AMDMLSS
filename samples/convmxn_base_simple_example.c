#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
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
    // Default to DEBUG level verbose mode
    MLSSenum verboseLevel = 4; // DEBUG level
    //MLSSstring asic = MLSS_GFXAUTOFIND;
    MLSSstring asic = MLSS_GFX1201;
    char customAsic[64] = {0};

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

    // Set the verbose level
    CHECK_STATUS(mlssSetVerboseLevel(verboseLevel));

    printf("=== CONV1x1 Simple Example ===\n");
    printf("Verbose level set to: %d\n\n", verboseLevel);

    MLSSuint32 w = 224;                  // Input x-dimension size.
    MLSSuint32 h = 224;                  // Input y-dimension size.
    MLSSuint32 c = 3;                    // Number of channels.
    MLSSuint32 n = 1;                    // Number of batches.
    MLSSuint32 k = 64;                   // Number of features.
    MLSSuint32 s = 7;                    // Filter x-dimension size.
    MLSSuint32 r = 7;                    // Filter y-dimension size.
    MLSSuint32 outW = 112;               // Output x-dimension size.
    MLSSuint32 outH = 112;               // Output y-dimension size.
    MLSSuint32 startPadX = 3;            // Zero padding added to the beginning of the input in the x-dimension.
    MLSSuint32 startPadY = 3;            // Zero padding added to the beginning of the input in the y-dimension.
    MLSSuint32 endPadX = 3;              // Zero padding added to the end of the input in the x-dimension.
    MLSSuint32 endPadY = 3;              // Zero padding added to the end of the input in the y-dimension.
    MLSSuint32 outPadX = 0;              // Zero padding added to the end of the output in the x-dimension.
    MLSSuint32 outPadY = 0;              // Zero padding added to the end of the output in the y-dimension.
    MLSSuint32 convStrideX = 2;          // Step between convolutions in the input x-dimension.
    MLSSuint32 convStrideY = 2;          // Step between convolutions in the input y-dimension.
    MLSSuint32 inputStrideX = 1;         // Step between dot products in the input x-dimension. Adds zero padding in the input.
    MLSSuint32 inputStrideY = 1;         // Step between dot products in the input y-dimension. Adds zero padding in the input.
    MLSSuint32 filterStrideX = 1;        // Step between dot products in the filter x-dimension. Adds zero padding in the filter.
    MLSSuint32 filterStrideY = 1;        // Step between dot products in the filter y-dimension. Adds zero padding in the filter.
    MLSSuint32 groups = 1;               // Split c and k into this many filter groups.
    MLSSbool   hasBias = true;           // If there is a bias tensor.
    MLSSbool   crossCorrelation = false; // If this is a cross correlation instead of a real convolution. Most ML convs are CCs.
    MLSSbool   backward = false;         // If this represents a "backward wrt inputs conv"/"transpose conv"/"deconvolution" layer.
                                         // The above parameters have already been adjusted to convert the backwards conv into an
                                         // equivalent forward conv. The meta command implementation must still swap its C and K
                                         // indices when accessing the filter (see swapCK in GenericConvResources).

    MLSSuint32 dNStride = 1;             // n stride of the input data
    MLSSuint32 dHStride = 1;             // h stride of the input data
    MLSSuint32 dCStride = 1;
    MLSSuint32 fKStride = 1;             // k stride
    MLSSuint32 fCStride = 1;             // c stride
    MLSSuint32 fRStride = 1;             // r stride
    MLSSuint32 fSStride = 1;             // s stride
    MLSSuint32 oNStride = 1;             // n stride of the output data
    MLSSuint32 oHStride = 1;             // h stride of the output data
    MLSSuint32 oKStride = 1;
    MLSSuint32 dOffset = 1;
    MLSSuint32 oOffset = 1;
    MLSSuint32 fOffset = 1;
    MLSSuint32 bOffset = 1;

    MLSSuint32 dataType = MLSS_FLOAT16;
    MLSSuint32 activation = MLSS_ACTIVATION_RELU;
    MLSSuint32 precision = MLSS_PRECISION_FLOAT16;

    MLSScontext context = 0;
    MLSSbinary* binaries = NULL;
    MLSSsize nn = 0;
    MLSSstring opName = MLSS_CONV;

    MLSSstatus* pStatuses = NULL;
    MLSSsize nStatuses = 0;

    // Step 1) Create context
    CHECK_STATUS(mlssCreateContext(&context, asic, opName));


    // Step 2) Set parameters
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_W, &w));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_H, &h));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_C, &c));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_N, &n));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_K, &k));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_S, &s));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_R, &r));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_OUTW, &outW));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_OUTH, &outH));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_STARTPADX, &startPadX));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_STARTPADY, &startPadY));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_ENDPADX, &endPadX));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_ENDPADY, &endPadY));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_OUTPADX, &outPadX));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_OUTPADY, &outPadY));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_CONVSTRIDEX, &convStrideX));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_CONVSTRIDEY, &convStrideY));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_INPUTSTRIDEX, &inputStrideX));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_INPUTSTRIDEY, &inputStrideY));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_FILTERSTRIDEX, &filterStrideX));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_FILTERSTRIDEY, &filterStrideY));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_GROUPS, &groups));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_HASBIAS, &hasBias));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_CROSSCORRELATION, &crossCorrelation));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_BACKWARD, &backward));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_DNSTRIDE, &dNStride));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_DHSTRIDE, &dHStride));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_DCSTRIDE, &dCStride));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_FKSTRIDE, &fKStride));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_FCSTRIDE, &fCStride));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_FRSTRIDE, &fRStride));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_FSSTRIDE, &fSStride));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_ONSTRIDE, &oNStride));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_OHSTRIDE, &oHStride));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_OKSTRIDE, &oKStride));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_DOFFSET, &dOffset));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_OOFFSET, &oOffset));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_FOFFSET, &fOffset));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_BOFFSET, &bOffset));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_DATATYPE, &dataType));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_PRECISION, &precision));
    CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_CONV_ACTIVATION, &activation));

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

    CHECK_STATUS(mlssGetBinaries(context, &binaries, &nn));

    CHECK_STATUS(mlssPrintBinaries(binaries, nn));

    printf("\n=== CONV1x1 Simple Example Complete ===\n");

    return EXIT_SUCCESS;
}
