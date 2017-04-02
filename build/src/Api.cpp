#define IMPLEMENT_API
#include <hx/CffiPrime.h>
#include <vector>

#include "c_api.h"

vkind tensorKind;

extern "C" void InitIDs()
{
    kind_share(&tensorKind,"tensor");
}

DEFINE_ENTRY_POINT(InitIDs)

extern "C" int tensorflux_register_prims()
{
   InitIDs();
   return 0;
}


#define TO_TENSOR \
   if (!val_is_kind(inTensor,tensorKind)) val_throw(alloc_string("object not a tensor")); \
   TF_Tensor *tensor = (TF_Tensor *)val_data(inTensor);


int tfGetDims(value inTensor)
{
   TO_TENSOR;
   return (int)TF_NumDims(tensor);
}
DEFINE_PRIME1(tfGetDims)


int tfGetDim(value inTensor,int inIndex)
{
   TO_TENSOR;
   return (int)TF_Dim(tensor,inIndex);
}
DEFINE_PRIME2(tfGetDim)


int tfGetByteSize(value inTensor)
{
   TO_TENSOR;
   return (int)TF_TensorByteSize(tensor);
}
DEFINE_PRIME1(tfGetByteSize)


void tfDestroy(value inTensor)
{
   TO_TENSOR;
   TF_DeleteTensor(tensor);
   val_gc(inTensor,0);
}
DEFINE_PRIME1v(tfDestroy)

const char *tfGetData(value inTensor)
{
   TO_TENSOR;
   return (const char *)TF_TensorData(tensor);
}
DEFINE_PRIME1(tfGetData)

void destroy_tensor(value inTensor)
{
   TO_TENSOR;
   if (tensor)
      TF_DeleteTensor(tensor);
}

value tfAllocate(int type, value dimArray, int byteCount)
{
   std::vector<int64_t> dims;
   int n = val_array_size(dimArray);
   for(int i=0;i<n;i++)
      dims.push_back( val_int(val_array_i(dimArray,i)) );
   TF_Tensor *tensor = TF_AllocateTensor((TF_DataType)type, &dims[0], dims.size(), byteCount);
   value result = alloc_abstract(tensorKind, tensor);
   val_gc(result, destroy_tensor);
   return result;
}
DEFINE_PRIME3(tfAllocate)


value tfAllocateInt32(int inValue)
{
   int64_t dim = 1;
   TF_Tensor *tensor = TF_AllocateTensor(TF_INT32, &dim, 1, sizeof(int));
   *(int *)TF_TensorData(tensor) = inValue;
   value result = alloc_abstract(tensorKind, tensor);
   val_gc(result, destroy_tensor);
   return result;
}
DEFINE_PRIME1(tfAllocateInt32)

