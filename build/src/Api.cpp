#define IMPLEMENT_API
#include <hx/CffiPrime.h>
#include <vector>

#include "c_api.h"

vkind tensorKind;
vkind contextKind;
vkind outputKind;

extern "C" void InitIDs()
{
   kind_share(&tensorKind,"tensor");
   kind_share(&contextKind,"tfContext");
   kind_share(&outputKind,"tfOutput");
}

DEFINE_ENTRY_POINT(InitIDs)

extern "C" int tensorflux_register_prims()
{
   InitIDs();
   return 0;
}


// ------ Tensor -----------------------



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
   TF_Tensor *tensor = TF_AllocateTensor(TF_INT32, &dim, 0, sizeof(int));
   *(int *)TF_TensorData(tensor) = inValue;
   value result = alloc_abstract(tensorKind, tensor);
   val_gc(result, destroy_tensor);
   return result;
}
DEFINE_PRIME1(tfAllocateInt32)


// --- Context -----------------------------------------

struct Context
{
   TF_OperationDescription *op;
   TF_Graph *graph;
   TF_Status* status;


   Context()
   {
      graph = TF_NewGraph();
      status = TF_NewStatus();
      op = 0;
   }
   ~Context()
   {
      TF_DeleteStatus(status);
      // Destroy an options object.  Graph will be deleted once no more
      // TFSession's are referencing it.
      TF_DeleteGraph(graph);
   }

   void beginOp(const char *inOp, const char *inName)
   {
      op = TF_NewOperation(graph, inOp, inName);
   }

   void checkStatus()
   {
      if (TF_GetCode(status)!=TF_OK)
         val_throw( alloc_string(TF_Message(status)));
   }

   TF_Output endForOutput()
   {
      TF_Operation *result = TF_FinishOperation(op,status);
      checkStatus();
      if (TF_OperationNumOutputs(result)!=1)
         val_throw( alloc_string("result does not match a single output"));
      return TF_Output(result,0);
   }

   void endForOutputArray(std::vector<TF_Output> &outputs)
   {
      TF_Operation *result = TF_FinishOperation(op,status);
      checkStatus();
      int n = TF_OperationNumOutputs(result);
      for(int i=0;i<n;i++)
         outputs.push_back( TF_Output(result,i) );
   }

};


#define TO_CONTEXT \
   if (!val_is_kind(inContext,contextKind)) val_throw(alloc_string("object not a context")); \
   Context *context = (Context *)val_data(inContext);



void destroy_context(value ctx) { delete (Context *)val_data(ctx); }


value ctxCreate()
{
   value result = alloc_abstract(contextKind, new Context());
   val_gc(result, destroy_context);
   return result;
}
DEFINE_PRIME0(ctxCreate)

void ctxBeginOp(value inContext, HxString opType, HxString name)
{
   TO_CONTEXT;
   context->beginOp(opType.c_str(), name.c_str());
}
DEFINE_PRIME3v(ctxBeginOp)

void ctxAddInput(value inContext, value inInput)
{
   TO_CONTEXT;
   if (!val_is_kind(inInput,outputKind)) val_throw(alloc_string("object not an output"));\
   TF_Output *output = (TF_Output *)val_data(inInput);
   TF_AddInput(context->op,*output);
}
DEFINE_PRIME2v(ctxAddInput)

void destroy_output(value o)
{
   delete (TF_Output *)val_data(o);
}

value ctxEndForOutput(value inContext)
{
   TO_CONTEXT;
   TF_Output *out = new TF_Output();
   *out = context->endForOutput();
   value result = alloc_abstract(outputKind, out);
   val_gc(result, destroy_output);
   return result;
}
DEFINE_PRIME1(ctxEndForOutput)

void ctxEndForOutputArray(value inContext,value outArray)
{
   TO_CONTEXT;
   std::vector<TF_Output> outputs;
   context->endForOutputArray(outputs);
   for(int i=0;i<outputs.size();i++)
   {
      TF_Output *out = new TF_Output();
      *out = outputs[i];
      value result = alloc_abstract(outputKind, out);
      val_gc(result, destroy_output);
      val_array_push(outArray, result);
   }
}
DEFINE_PRIME2v(ctxEndForOutputArray)

void ctxAddAttribInt(value inContext, HxString inName, int inValue)
{
   TO_CONTEXT;
   TF_SetAttrInt(context->op, inName.c_str(), inValue);
}
DEFINE_PRIME3v(ctxAddAttribInt)


void ctxAddAttribFloat(value inContext, HxString inName, double inValue)
{
   TO_CONTEXT;
   TF_SetAttrFloat(context->op, inName.c_str(), (float)inValue);
}
DEFINE_PRIME3v(ctxAddAttribFloat)

void ctxAddAttribType(value inContext, HxString inName, int inValue)
{
   TO_CONTEXT;
   TF_SetAttrType(context->op, inName.c_str(), (TF_DataType)inValue);
}
DEFINE_PRIME3v(ctxAddAttribType)


void ctxAddAttribTensor(value inContext, HxString inName, value inTensor)
{
   TO_CONTEXT;
   TO_TENSOR;
   TF_SetAttrTensor(context->op, inName.c_str(), tensor,context->status);
   context->checkStatus();
}
DEFINE_PRIME3v(ctxAddAttribTensor)

