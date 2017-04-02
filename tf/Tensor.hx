package tf;

import Type as GlobalType;

abstract Tensor(Dynamic)
{
   public var dims(get,never):Int;
   public var byteSize(get,never):Int;

   public function new(d:Dynamic) this = d;
   public function getDim(index:Int) return Api.tfGetDim(this, index);

   public function get_dims():Int return Api.tfGetDims(this);
   public function get_byteSize():Int return Api.tfGetByteSize(this);

   inline public function destroy():Void { Api.tfDestroy(this); this=null; }

   #if cpp
   public var data(get,never):cpp.Pointer<cpp.Void>;
   public function get_data():cpp.Pointer<cpp.Void> return cast Api.tfGetData(this);
   #end

   public static function allocate(type:tf.Type, dims:Array<Int>, byteCount:Int) : Tensor
   {
      return new Tensor( Api.tfAllocate( GlobalType.enumIndex(type), dims, byteCount ) );
   }

   public static function int32(inVal:Int) : Tensor
   {
      return new Tensor( Api.tfAllocateInt32(inVal) );
   }
}
