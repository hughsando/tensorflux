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

   public var dataHandle(get,never):Dynamic;
   public function get_dataHandle():Dynamic return tfGetDataHandle(this);

   #if cpp
   public var data(get,never):cpp.Pointer<cpp.Void>;
   public function get_data():cpp.Pointer<cpp.Void> return cast tfGetData(this);
   #end

   public static function allocate(type:tf.Type, dims:Array<Int>, byteCount:Int) : Tensor
   {
      return new Tensor( tfAllocate( GlobalType.enumIndex(type), dims, byteCount ) );
   }

   @:from
   public static function int32(inVal:Int) : Tensor
   {
      return new Tensor( tfAllocateInt32(inVal) );
   }

   @:from
   public static function fromBool(inVal:Bool) : Tensor
   {
      return new Tensor( tfAllocateBool(inVal) );
   }


   @:from
   public static function float(inVal:Float) : Tensor
   {
      return new Tensor( tfAllocateFloat(inVal) );
   }


   // Todo: restruct to Array<Float>, Array<Float32> etc.
   public static function floats(value:Dynamic, ?shape:Array<Int>) : Tensor
   {
      return new Tensor( tfAllocateFloats(value,shape) );
   }

   public static function ints(value:Dynamic, ?shape:Array<Int>) : Tensor
   {
      return new Tensor( tfAllocateInts(value,shape) );
   }


   public static function int64s(value:Dynamic, ?shape:Array<Int>) : Tensor
   {
      return new Tensor( tfAllocateInt64s(value,shape) );
   }


   public static function bytes(value:Dynamic, ?shape:Array<Int>) : Tensor
   {
      return new Tensor( tfAllocateBytes(value,shape) );
   }

   public function toString()
   {
      return tfToString(this);
   }


   public static var tfGetDims = Loader.load("tfGetDims","oi");
   public static var tfGetDim = Loader.load("tfGetDim","oii");
   public static var tfGetByteSize = Loader.load("tfGetByteSize","oi");
   public static var tfDestroy = Loader.load("tfDestroy","ov");
   public static var tfGetDataHandle = Loader.load("tfGetDataHandle","oo");
   #if cpp
   public static var tfGetData = Loader.load("tfGetData","oc");
   #end
   public static var tfAllocate = Loader.load("tfAllocate","ioio");
   public static var tfAllocateInt32 = Loader.load("tfAllocateInt32","io");
   public static var tfAllocateFloat = Loader.load("tfAllocateFloat","do");
   public static var tfAllocateBool = Loader.load("tfAllocateBool","bo");
   public static var tfAllocateFloats = Loader.load("tfAllocateFloats","ooo");
   public static var tfAllocateInts = Loader.load("tfAllocateInts","ooo");
   public static var tfAllocateInt64s = Loader.load("tfAllocateInt64s","ooo");
   public static var tfAllocateBytes = Loader.load("tfAllocateBytes","ooo");
   public static var tfToString = Loader.load("tfToString","os");

}
