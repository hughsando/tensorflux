#define IMPLEMENT_API
#include <hx/CffiPrime.h>
#include <vector>



#include "c_api.h"

vkind tensorKind;
vkind contextKind;
vkind outputKind;
vkind sessionKind;

extern "C" void InitIDs()
{
   kind_share(&tensorKind,"tensor");
   kind_share(&contextKind,"tfContext");
   kind_share(&outputKind,"tfOutput");
   kind_share(&sessionKind,"tfSession");
}

DEFINE_ENTRY_POINT(InitIDs)

extern "C" int tensorflux_register_prims()
{
   InitIDs();
   return 0;
}

#define TO_OUTPUT(X) \
   if (!val_is_kind(X,outputKind)) val_throw(alloc_string("object not an output")); \
   TF_Output *output = (TF_Output *)val_data(X);

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
   int n = val_array_size(dimArray);
   std::vector<int64_t> dims(n);
   for(int i=0;i<n;i++)
      dims[0] = val_int(val_array_i(dimArray,i));
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


value tfAllocateFloat(double inValue)
{
   int64_t dim = 1;
   TF_Tensor *tensor = TF_AllocateTensor(TF_FLOAT, &dim, 0, sizeof(int));
   *(float *)TF_TensorData(tensor) = (float)inValue;
   value result = alloc_abstract(tensorKind, tensor);
   val_gc(result, destroy_tensor);
   return result;
}
DEFINE_PRIME1(tfAllocateFloat)


template<typename T>
value tfAllocateArray(value inData, value inShape,TF_DataType inType)
{
   int64_t n = 0;
   bool isArray = true;
   if (!val_is_null(inData))
   {
      n = val_array_size(inData);
      if (n==0 && val_is_float(inData))
      {
         n = 1;
         isArray = false;
      }
   }
   std::vector<int64_t> dimVals;
   // default to scalar
   int dimCount = 0;
   if (val_is_null(inShape))
   {
      // 1-D array from data
      if (!isArray)
      {
         dimCount = 1;
         dimVals.push_back(n);
      }
   }
   else
   {
      dimCount = val_array_size(inShape);
      for(int i=0;i<dimCount;i++)
         dimVals.push_back( val_int( val_array_i(inShape,i) ) );
   }

   size_t elements = 1;
   if (dimCount==0)
   {
      dimVals.push_back(0);
   }
   else
   {
      int missingDim = -1;
      for(int i=0;i<dimCount;i++)
      {
         if (dimVals[i]<=0)
         {
            if (missingDim>=0)
               val_throw(alloc_string("Too many unspecified dimensions"));
            missingDim = i;
         }
         else
         {
            elements *= dimVals[i];
         }
      }
      if (missingDim<0)
      {
         size_t val = (n+elements-1) / elements;
         dimVals[missingDim] = val;
         elements *= val;
      }
   }


   TF_Tensor *tensor = TF_AllocateTensor(inType, &dimVals[0], dimCount, elements*sizeof(T));
   T *data = (T *)TF_TensorData(tensor);
   if (n>elements)
      n = elements;
   if (n<1)
   {
      // huh?
   }
   else if (isArray)
   {
      float *f=0;
      double *d=0;
      bool *b=0;
      int *I=0;
      value *v=0;

      if ( (f=val_array_float(inData)))
      {
         if (inType==TF_FLOAT)
            memcpy(data, f, sizeof(float)*n );
         else
            for(int i=0;i<n;i++)
               data[i] = f[i];
      }
      else if ( (d=val_array_double(inData)) )
      {
         for(int i=0;i<n;i++)
            data[i] = d[i];
      }
      else if ( (I=val_array_int(inData)) )
      {
         if (inType==TF_INT32)
            memcpy(data, I, sizeof(int)*n );
         else
            for(int i=0;i<n;i++)
               data[i] = I[i];
      }
      else if ( (b=val_array_bool(inData)) )
      {
         for(int i=0;i<n;i++)
            data[i] = b[i];
      }
      else if ( (v=val_array_value(inData)) )
      {
         if (inType==TF_FLOAT)
            for(int i=0;i<n;i++)
               data[i] = val_float(v[i]);
         else
            for(int i=0;i<n;i++)
               data[i] = val_int(v[i]);
      }
      else
      {
         if (inType==TF_FLOAT)
            for(int i=0;i<n;i++)
               data[i] = val_float(val_array_i(inData,i));
         else
            for(int i=0;i<n;i++)
               data[i] = val_int(val_array_i(inData,i));
      }

      if (n<elements)
         memset(data + n, 0, (elements-n)*sizeof(T));
   }
   else
   {
      if (inType==TF_FLOAT)
         *data = (T)val_float(inData);
      else
         *data = (T)val_int(inData);
   }

   value result = alloc_abstract(tensorKind, tensor);
   val_gc(result, destroy_tensor);
   return result;
}

value tfAllocateFloats(value inData, value inShape)
{
   return tfAllocateArray<float>(inData, inShape, TF_FLOAT);
}
DEFINE_PRIME2(tfAllocateFloats)


value tfAllocateInts(value inData, value inShape)
{
   return tfAllocateArray<int>(inData, inShape, TF_INT32);
}
DEFINE_PRIME2(tfAllocateInts)


value tfAllocateInt64s(value inData, value inShape)
{
   return tfAllocateArray<int64_t>(inData, inShape, TF_INT64);
}
DEFINE_PRIME2(tfAllocateInt64s)


value tfAllocateBytes(value inData, value inShape)
{
   return tfAllocateArray<int64_t>(inData, inShape, TF_UINT8);
}
DEFINE_PRIME2(tfAllocateBytes)


HxString tfToString(value inTensor)
{
   TO_TENSOR;
   std::string result;
   TF_TensorToString(tensor,result);
   return HxString(result.c_str(), result.size());
}
DEFINE_PRIME1(tfToString)


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
   TO_OUTPUT(inInput);
   TF_AddInput(context->op,*output);
}
DEFINE_PRIME2v(ctxAddInput)


void ctxAddInputArray(value inContext, value inInputs)
{
   TO_CONTEXT;
   int n = val_array_size(inInputs);
   if (!n) return;

   std::vector<TF_Output> inputs(n);
   for(int i=0;i<n;i++)
   {
      value input = val_array_i(inInputs,i);
      TO_OUTPUT(input);
      inputs[i] = *output;
   }
   TF_AddInputList(context->op,&inputs[0],n);
}
DEFINE_PRIME2v(ctxAddInputArray)



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


void ctxAddAttribIntArray(value inContext, HxString inName, value inValue)
{
   TO_CONTEXT;
   int n = val_array_size(inValue);
   if (!n) return;

   std::vector<int64_t> values(n);
   int *iPtr = val_array_int(inValue);
   if (iPtr)
   {
      for(int i=0;i<n;i++)
         values[i] = iPtr[i];
   }
   else
   {
      for(int i=0;i<n;i++)
         values[i] = val_int(val_array_i(inValue,i));
   }
   TF_SetAttrIntList(context->op, inName.c_str(), &values[0], n);
}
DEFINE_PRIME3v(ctxAddAttribIntArray)


void ctxAddAttribStringArray(value inContext, HxString inName, value inValue)
{
   TO_CONTEXT;
   int n = val_array_size(inValue);
   if (!n) return;

   std::vector<std::string> values(n);
   std::vector<const char *> pointers(n);
   std::vector<size_t> lengths(n);
   for(int i=0;i<n;i++)
   {
      value s = val_array_i(inValue,i);
      int len = val_strlen(s);
      const char *str = val_string(s);
      values[i] = std::string(str,str+len);
      pointers[i] = values[i].c_str();
      lengths[i] = len;
   }
   TF_SetAttrStringList(context->op, inName.c_str(), (const void* const*)&pointers[0], &lengths[0], n);
}
DEFINE_PRIME3v(ctxAddAttribStringArray)



void ctxAddAttribShape(value inContext, HxString inName, value inValue)
{
   TO_CONTEXT;
   int n = val_array_size(inValue);
   if (!n) return;

   std::vector<int64_t> values(n);
   int *iPtr = val_array_int(inValue);
   if (iPtr)
   {
      for(int i=0;i<n;i++)
         values[i] = iPtr[i];
   }
   else
   {
      for(int i=0;i<n;i++)
         values[i] = val_int(val_array_i(inValue,i));
   }
   TF_SetAttrShape(context->op, inName.c_str(), &values[0], n);
}
DEFINE_PRIME3v(ctxAddAttribShape)



void ctxAddAttribTypeList(value inContext, HxString inName, value inValue)
{
   TO_CONTEXT;
   int n = val_array_size(inValue);
   if (!n) return;

   std::vector<TF_DataType> values(n);
   int *iPtr = val_array_int(inValue);
   if (iPtr)
   {
      for(int i=0;i<n;i++)
         values[i] = (TF_DataType)iPtr[i];
   }
   else
   {
      for(int i=0;i<n;i++)
         values[i] = (TF_DataType)val_int(val_array_i(inValue,i));
   }
   TF_SetAttrTypeList(context->op, inName.c_str(), &values[0], n);
}
DEFINE_PRIME3v(ctxAddAttribTypeList)



void ctxAddAttribFloatArray(value inContext, HxString inName, value inValue)
{
   TO_CONTEXT;
   int n = val_array_size(inValue);
   if (!n) return;

   std::vector<float> values(n);
   float *fPtr = val_array_float(inValue);
   if (fPtr)
   {
      for(int i=0;i<n;i++)
         values[i] = fPtr[i];
   }
   else
   {
      double *dPtr = val_array_double(inValue);
      if (dPtr)
      {
         for(int i=0;i<n;i++)
            values[i] = dPtr[i];
      }
      else
         for(int i=0;i<n;i++)
            values[i] = val_float(val_array_i(inValue,i));
   }
   TF_SetAttrFloatList(context->op, inName.c_str(), &values[0], n);
}
DEFINE_PRIME3v(ctxAddAttribFloatArray)



void ctxAddAttribBool(value inContext, HxString inName, bool inValue)
{
   TO_CONTEXT;
   TF_SetAttrBool(context->op, inName.c_str(), inValue);
}
DEFINE_PRIME3v(ctxAddAttribBool)


void ctxAddAttribString(value inContext, HxString inName,HxString inValue)
{
   TO_CONTEXT;
   TF_SetAttrString(context->op, inName.c_str(), inValue.c_str(),inValue.length);
}
DEFINE_PRIME3v(ctxAddAttribString)


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

// --- Session ---------------------------------------------------

value sesCreate(value inContext, value inConfig, HxString target)
{
   TO_CONTEXT;

   TF_SessionOptions *opts = TF_NewSessionOptions();
   if (!val_is_null(inConfig))
   {
      tensorflow::ConfigProto config;
      value cpuCount = val_field(inConfig, val_id("cpuCount"));
      if (!val_is_null(cpuCount))
         (*config.mutable_device_count())["CPU"] = val_int(cpuCount);
      value gpuCount = val_field(inConfig, val_id("gpuCount"));
      if (!val_is_null(gpuCount))
         (*config.mutable_device_count())["GPU"] = val_int(gpuCount);
      value logPlace = val_field(inConfig, val_id("logDevicePlacement"));
      if (!val_is_null(logPlace))
         config.set_log_device_placement(val_bool(logPlace));
      TF_SetConfig(opts, config);
   }

   TF_Session *session = TF_NewSession(context->graph, opts, context->status);
   TF_DeleteSessionOptions(opts);

   context->checkStatus();

   value result = alloc_abstract(sessionKind, session);
   //val_gc(result, destroy_session);
   return result;
}
DEFINE_PRIME3(sesCreate)

#define TO_SESSION \
   if (!val_is_kind(inSession,sessionKind)) val_throw(alloc_string("object not a session")); \
   TF_Session *session = (TF_Session *)val_data(inSession);

void sesClose(value inSession)
{
   TO_SESSION;

   TF_Status *status = TF_NewStatus();
   TF_CloseSession(session,status);
   // TODO - separate?
   if (TF_GetCode(status)==TF_OK)
      TF_DeleteSession(session,status);

   if (TF_GetCode(status)!=TF_OK)
   {
      value err = alloc_string(TF_Message(status));
      TF_DeleteStatus(status);
      val_throw(err);
   }
   TF_DeleteStatus(status);
}
DEFINE_PRIME1v(sesClose)


void sesRun(value inSession, value fetches, value feedTargets, value feedValues, value result)
{
   TO_SESSION;

   int outputCount = val_array_size(fetches);
   int inputCount = val_is_null(feedTargets) ? 0 :  val_array_size(feedTargets);

   std::vector<TF_Output> inputs(inputCount);
   std::vector<TF_Tensor *> inputValues(inputCount);
   std::vector<TF_Output> outputs(outputCount);
   std::vector<TF_Tensor *> resultTensors(outputCount);

   for(int i=0;i<inputCount;i++)
   {
      value in = val_array_i(feedTargets,i);
      TO_OUTPUT(in);
      inputs[i] = *output;
      value inTensor = val_array_i(feedValues,i);
      TO_TENSOR;
      inputValues[i] = tensor;
   }

   for(int i=0;i<outputCount;i++)
   {
      value out = val_array_i(fetches,i);
      TO_OUTPUT(out);
      outputs[i] = *output;
   }

   TF_Status *status = TF_NewStatus();
   TF_SessionRun(session,
                 0, // RunOptions const TF_Buffer* run_options,
                 // Input tensors
                 &inputs[0],
                 &inputValues[0],
                 inputCount,
                 // Output tensors
                 &outputs[0],
                 &resultTensors[0],
                 outputCount,
                 // Target operations
                 0,0, //const TF_Operation* const* target_opers, int ntargets,
                 0, // TF_Buffer* run_metadata,
                 // Output status
                 status);
   if (TF_GetCode(status)!=TF_OK)
   {
      value err = alloc_string(TF_Message(status));
      TF_DeleteStatus(status);
      val_throw(err);
   }
   TF_DeleteStatus(status);

   for(int i=0;i<outputCount;i++)
   {
      value tval = alloc_abstract(tensorKind, resultTensors[i]);
      val_gc(tval, destroy_tensor);
      val_array_set_i(result, i, tval);
   }
}
DEFINE_PRIME5v(sesRun)
