package tf;

import Type as GlobalType;

abstract Tensor(Dynamic)
{
   public var dims(get,never):Int;
   public var byteSize(get,never):Int;

   public function new(d:Dynamic) this = d;
   public function getDim(index:Int) return tfGetDim(this, index);

   public function get_dims():Int return tfGetDims(this);
   public function get_byteSize():Int return tfGetByteSize(this);

   inline public function destroy():Void { tfDestroy(this); this=null; }

   #if cpp
   public var data(get,never):cpp.Pointer<cpp.Void>;
   public function get_data():cpp.Pointer<cpp.Void> return cast tfGetData(this);
   #end

   public static function allocate(type:tf.Type, dims:Array<Int>, byteCount:Int) : Tensor
   {
      return new Tensor( tfAllocate( GlobalType.enumIndex(type), dims, byteCount ) );
   }

   public static function int32(inVal:Int) : Tensor
   {
      return new Tensor( tfAllocateInt32(inVal) );
   }
   public function toString()
   {
      return tfToString(this);
   }


   public static var tfGetDims = Loader.load("tfGetDims","oi");
   public static var tfGetDim = Loader.load("tfGetDim","oii");
   public static var tfGetByteSize = Loader.load("tfGetByteSize","oi");
   public static var tfDestroy = Loader.load("tfDestroy","ov");
   #if cpp
   public static var tfGetData = Loader.load("tfGetData","oc");
   #end
   public static var tfAllocate = Loader.load("tfAllocate","ioio");
   public static var tfAllocateInt32 = Loader.load("tfAllocateInt32","io");
   public static var tfToString = Loader.load("tfToString","os");

}
