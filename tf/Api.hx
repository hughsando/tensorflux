package tf;

class Api
{
   public static var tfGetDims = Loader.load("tfGetDim","oi");
   public static var tfGetDim = Loader.load("tfGetDim","oii");
   public static var tfGetByteSize = Loader.load("tfGetByteSize","oi");
   public static var tfDestroy = Loader.load("tfDestroy","ov");
   #if cpp
   public static var tfGetData = Loader.load("tfGetData","oc");
   #end
   public static var tfAllocate = Loader.load("tfAllocate","ioio");
   public static var tfAllocateInt32 = Loader.load("tfAllocateInt32","io");
}
