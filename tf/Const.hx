package tf;

class Const
{
   public static function int32(?inName:String,value:Int) : Output
   {
      var tensor = Tensor.int32(value);
      return ConstOps.const(inName, tensor, Type.Int32 );
   }

   // Todo: restruct to Array<Float>, Array<Float32> etc.
   public static function floats(?inName:String,value:Dynamic, ?shape:Array<Int>) : Output
   {
      var tensor = Tensor.floats(value,shape);
      return ConstOps.const(inName, tensor, Type.Float32 );
   }

   public static function ints(?inName:String,value:Dynamic, ?shape:Array<Int>) : Output
   {
      var tensor = Tensor.ints(value,shape);
      return ConstOps.const(inName, tensor, Type.Int32 );
   }

   public static function int64s(?inName:String,value:Dynamic, ?shape:Array<Int>) : Output
   {
      var tensor = Tensor.int64s(value,shape);
      return ConstOps.const(inName, tensor, Type.Int64 );
   }

   public static function bytes(?inName:String,value:Dynamic, ?shape:Array<Int>) : Output
   {
      var tensor = Tensor.bytes(value,shape);
      return ConstOps.const(inName, tensor, Type.Int8 );
   }

}

