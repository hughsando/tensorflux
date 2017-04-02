package tf;

class Const
{
   public static function int32(?inName:String,value:Int) : Output
   {
      var tensor = Tensor.int32(value);
      return ConstOps.Const(inName, tensor, Type.Int32 );
   }
}

