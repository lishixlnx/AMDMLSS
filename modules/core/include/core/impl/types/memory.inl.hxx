#pragma once

namespace mlss
{
    //=====================================================================================================================
    //                                   ObjectManager
    //=====================================================================================================================

    //---------------------------------------------------------------------
    template <class ObjectType>
    typename ObjectManager<ObjectType>::handle_type
    ObjectManager<ObjectType>::addObject(object_type&& object)
    {

        auto ptr = std::make_unique<typename std::decay_t<object_type>>(std::move(object));
        return addObject(std::move(ptr));
    }

    //---------------------------------------------------------------------
    template <class ObjectType>
    typename ObjectManager<ObjectType>::handle_type
    ObjectManager<ObjectType>::addObject(std::unique_ptr<object_type>&& object)
    {
        auto& instance = createInstance();
        std::lock_guard lock(instance.m_mutex);

        void* raw_ptr = object.get();
        instance.m_objects.insert(std::move(object));

        return reinterpret_cast<handle_type>(raw_ptr);
    }

    //---------------------------------------------------------------------
    template <class ObjectType>
    template <typename T>
    T* ObjectManager<ObjectType>::getPointer(handle_type handle)
    {
        return reinterpret_cast<T*>(handle);
    }

    //---------------------------------------------------------------------
    template <class ObjectType>
    template <typename PtrType>
    bool ObjectManager<ObjectType>::isInitialized(PtrType* ptr)
    {
        auto& instance = createInstance();
        std::lock_guard lock(instance.m_mutex);
        return instance.isInitialized_(ptr);
    }

    //---------------------------------------------------------------------
    template <class ObjectType>
    template <typename PtrType>
    void ObjectManager<ObjectType>::markAsInitialized(PtrType* ptr)
    {
        auto& instance = createInstance();
        std::lock_guard lock(instance.m_mutex);
        instance.markAsInitialized_(ptr);
    }

    //---------------------------------------------------------------------
    template <class ObjectType>
    template <typename PtrType>
    bool ObjectManager<ObjectType>::isInitialized_(PtrType* ptr) const
    {
        if (!ptr)
        {
            return false;
        }

        return m_initialized_ptr.find(static_cast<const void*>(ptr)) != m_initialized_ptr.end();
    }

    //---------------------------------------------------------------------
    template <class ObjectType>
    template <typename PtrType>
    void ObjectManager<ObjectType>::markAsInitialized_(PtrType* ptr)
    {
        if (ptr)
        {
            m_initialized_ptr.insert(static_cast<const void*>(ptr));
        }
    }

    //---------------------------------------------------------------------
    template <class ObjectType>
    ObjectManager<ObjectType>& ObjectManager<ObjectType>::createInstance()
    {
        static ObjectManager instance;
        return instance;
    }

} // namespace mlss
