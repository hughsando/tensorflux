package tf;

class Context
{
   // TLS ?
   static var currentContext:Context;
   var handle:Dynamic;

   public static var current(get,null):Dynamic;

   function new()
   {
      handle = null;
   }

   public static function get_current() : Context
   {
      if (currentContext==null)
         currentContext = new Context();
      return currentContext;
   }

   public function beginOp(opName:String, nodeName:String)
   {
   }
   public function addInput(i:tf.Tensor)
   {
   }
   public function endForOutput() : tf.Tensor
   {
      return null;
   }
   public function endForOutputArray() : Array<tf.Tensor>
   {
      return null;
   }

   public function addAttribInt(name:String, value:Int):Void
   {
   }

   public function addAttribFloat(name:String, value:Float):Void
   {
   }

   public function addAttribType(name:String, value:tf.Type):Void
   {
   }


   public function addAttribTensor(name:String, value:tf.Type):Void
   {
   }

   public function get_bad_color() : tf.Tensor { return null; }


}

