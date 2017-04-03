package tf;

abstract Output(Dynamic)
{
   inline public function new(v:Dynamic) this = v;

   @:from
   static public function fromInt32(i:Int) return Const.int32(i);


   @:op(A * B)
   public function multiply(right:Output) : Output
   {
      return MathOps.multiply(null, cast this,right);
   }

   @:op(A + B)
   public function add(right:Output) : Output
   {
      return MathOps.add(null, cast this,right);
   }

   @:op(A - B)
   public function subtract(right:Output) : Output
   {
      return MathOps.subtract(null, cast this,right);
   }

   @:op(A / B)
   public function divide(right:Output) : Output
   {
      return MathOps.realDiv(null, cast this,right);
   }


}

