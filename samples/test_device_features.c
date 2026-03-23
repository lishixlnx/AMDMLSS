#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "amdmlss/amdmlss_api.h"

void printDeviceFeatures(const MLSSdevicefeatures* device, int index)
{
    printf("Device %d:\n", index);
    printf("  GFX Architecture: %s\n", device->m_gfx ? device->m_gfx : "Unknown");
    printf("  Is Dedicated: %s\n", device->m_isDedicated ? "Yes" : "No");
    printf("\n");
}

int main()
{
    MLSSstatus status;
    MLSSdevicefeatures* devices = NULL;
    MLSSsize numDevices = 0;
    
    printf("AMDMLSS Device Query Example\n");
    printf("============================\n\n");
    
    // Query all available devices
    printf("Querying all available devices...\n");
    status = mlssGetDeviceFeatures(&devices, &numDevices);
    
    if (status != MLSS_SUCCESS)
    {
        printf("Error: Failed to get device features (status: %d)\n", status);
        return 1;
    }
    
    if (numDevices == 0)
    {
        printf("No devices found.\n");
        return 0;
    }
    
    printf("Found %zu device(s):\n\n", (size_t)numDevices);
    
    // Print information about all devices
    for (MLSSsize i = 0; i < numDevices; i++)
    {
        printDeviceFeatures(&devices[i], (int)i);
    }
    
    // Query the optimal device
    printf("Querying optimal device...\n");
    MLSSdevicefeatures optimalDevice;
    status = mlssGetOptimalDeviceFeatures(&optimalDevice);
    
    if (status == MLSS_SUCCESS)
    {
        printf("\nOptimal device selected:\n");
        printf("  GFX Architecture: %s\n", optimalDevice.m_gfx ? optimalDevice.m_gfx : "Unknown");
        printf("  Is Dedicated: %s\n", optimalDevice.m_isDedicated ? "Yes" : "No");
        
        // Note: The string in optimalDevice.m_gfx is owned by the library
        // and should not be freed by the user
    }
    else
    {
        printf("Error: Failed to get optimal device features (status: %d)\n", status);
    }
    
    // No need to free memory - it's managed by the MemoryManager
    
    printf("\nDevice query example completed.\n");
    return 0;
}
