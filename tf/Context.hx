package tf;

import Type as GlobalType;

class Context
{
   // TLS ?
   static var currentContext:Context;
   var handle:Dynamic;
   var scopeStack:Array< Scope >;

   public static var current(get,null):Dynamic;

   function new()
   {
      handle = ctxCreate();
      scopeStack = [ new Scope("") ];
   }

   public static function get_current() : Context
   {
      if (currentContext==null)
         currentContext = new Context();
      return currentContext;
   }

   public function beginOp(opName:String, nodeName:String) : Void
   {
      if (nodeName==null)
         nodeName = opName;
      nodeName = scopeStack[scopeStack.length-1].addUnique(nodeName);
      ctxBeginOp(handle,opName, nodeName);
   }
   public function addInput(i:Output) : Void
   {
      ctxAddInput(handle,i);
   }
   public function endForOutput() : Output
   {
      return ctxEndForOutput(handle);
   }
   public function endForOutputArray() : Array<Output>
   {
      var result = new Array<Output>();
      ctxEndForOutputArray(handle,result);
      return result;
   }

   public function addAttribInt(name:String, value:Int):Void
   {
      ctxAddAttribInt(handle, name, value);
   }

   public function addAttribFloat(name:String, value:Float):Void
   {
      ctxAddAttribFloat(handle, name, value);
   }

   public function addAttribType(name:String, value:tf.Type):Void
   {
      ctxAddAttribType(handle, name, GlobalType.enumIndex(value));
   }

   public function addAttribTensor(name:String, value:Tensor):Void
   {
      ctxAddAttribTensor(handle, name, value);
   }

   public function get_bad_color() : tf.Tensor { return null; }



   static var ctxCreate = Loader.load("ctxCreate","o");
   static var ctxBeginOp = Loader.load("ctxBeginOp","ossv");
   static var ctxAddInput = Loader.load("ctxAddInput","oov");
   static var ctxEndForOutput = Loader.load("ctxEndForOutput","oo");
   static var ctxEndForOutputArray = Loader.load("ctxEndForOutputArray","oov");
   static var ctxAddAttribInt = Loader.load("ctxAddAttribInt","osiv");
   static var ctxAddAttribFloat = Loader.load("ctxAddAttribFloat","osdv");
   static var ctxAddAttribType = Loader.load("ctxAddAttribType","osiv");
   static var ctxAddAttribTensor = Loader.load("ctxAddAttribTensor","osov");

}

