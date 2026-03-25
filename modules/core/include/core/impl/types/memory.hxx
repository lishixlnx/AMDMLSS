#pragma once



#include <cstdint>
namespace mlss
{
    // using booltype = bool;  // Replaced with bool

    //=================================================================================================================
    //                                   ObjectManager                                                                   
    //=================================================================================================================
    template<class ObjectType>
    class ObjectManager
    {
    public:
        using object_type = ObjectType;
        using pointer = object_type*;
        using handle_type = std::uintptr_t;  // For storing addresses  

        //---------------------------------------------------------------------
        ~ObjectManager() = default;

        //---------------------------------------------------------------------
        static handle_type addObject(object_type&& object);

        //---------------------------------------------------------------------
        static handle_type addObject(std::unique_ptr<object_type>&& object);

        //---------------------------------------------------------------------
        template<typename T = object_type>
        static T* getPointer(handle_type handle);

        //---------------------------------------------------------------------
        template<typename PtrType>
        static bool isInitialized(PtrType* ptr);

        //---------------------------------------------------------------------
        template<typename PtrType>
        static void markAsInitialized(PtrType* ptr);

    private:

        //---------------------------------------------------------------------
        static ObjectManager& createInstance();

        //---------------------------------------------------------------------
        template<typename PtrType>
        bool isInitialized_(PtrType* ptr) const;

        //---------------------------------------------------------------------
        template<typename PtrType>
        void markAsInitialized_(PtrType* ptr);

        //---------------------------------------------------------------------
        std::unordered_set<std::unique_ptr<object_type>> m_objects;
        std::unordered_set<const void*> m_initialized_ptr;  // Store void* for any pointer level  
        std::mutex m_mutex;
    };

    //=================================================================================================================
    using MemoryManager = ObjectManager<Any>;

} // mlss
