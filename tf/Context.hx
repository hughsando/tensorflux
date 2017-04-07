package tf;

import Type as GlobalType;

class Context
{
   // TLS ?
   static var currentContext:Context;
   var handle:Dynamic;
   var scopeStack:Array< Scope >;

   public static var current(get,null):Context;
   public static var currentHandle(get,null):Dynamic;

   function new(inVerbose = false)
   {
      handle = ctxCreate(inVerbose);
      scopeStack = [ new Scope("") ];
      currentContext = this;
   }

   public static function get_current() : Context
   {
      if (currentContext==null)
         new Context();
      return currentContext;
   }

   public static function get_currentHandle() : Dynamic
   {
      return current.handle;
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
   public function addInputArray(inputs:Array<Output>) : Void
   {
      ctxAddInputArray(handle,inputs);
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

   public function addAttribIntArray(name:String, value:Array<Int>):Void
   {
      ctxAddAttribIntArray(handle, name, value);
   }
   public function addAttribShape(name:String, value:Array<Int>):Void
   {
      ctxAddAttribShape(handle, name, value);
   }


   public function addAttribTypeArray(name:String, value:Array<Type>):Void
   {
      var ints = new Array<Int>();
      for(t in value)
        ints.push( GlobalType.enumIndex(t) );
      ctxAddAttribTypeArray(handle, name, ints);
   }

   public function addAttribFloatArray(name:String, value:Array<Float>):Void
   {
      ctxAddAttribFloatArray(handle, name, value);
   }
   public function addAttribInt(name:String, value:Int):Void
   {
      ctxAddAttribInt(handle, name, value);
   }

   public function addAttribFloat(name:String, value:Float):Void
   {
      ctxAddAttribFloat(handle, name, value);
   }

   public function addAttribString(name:String, value:String):Void
   {
      ctxAddAttribString(handle, name, value);
   }

   public function addAttribStringArray(name:String, value:Array<String>):Void
   {
      ctxAddAttribStringArray(handle, name, value);
   }

   public function addAttribBool(name:String, value:Bool):Void
   {
      ctxAddAttribBool(handle, name, value);
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



   static var ctxCreate = Loader.load("ctxCreate","bo");
   static var ctxBeginOp = Loader.load("ctxBeginOp","ossv");
   static var ctxAddInput = Loader.load("ctxAddInput","oov");
   static var ctxAddInputArray = Loader.load("ctxAddInputArray","oov");
   static var ctxEndForOutput = Loader.load("ctxEndForOutput","oo");
   static var ctxEndForOutputArray = Loader.load("ctxEndForOutputArray","oov");
   static var ctxAddAttribInt = Loader.load("ctxAddAttribInt","osiv");
   static var ctxAddAttribIntArray = Loader.load("ctxAddAttribIntArray","osov");
   static var ctxAddAttribFloat = Loader.load("ctxAddAttribFloat","osdv");
   static var ctxAddAttribFloatArray = Loader.load("ctxAddAttribIntArray","osov");
   static var ctxAddAttribShape = Loader.load("ctxAddAttribShape","osov");
   static var ctxAddAttribTypeArray = Loader.load("ctxAddAttribIntArray","osov");
   static var ctxAddAttribType = Loader.load("ctxAddAttribType","osiv");
   static var ctxAddAttribBool = Loader.load("ctxAddAttribBool","osbv");
   static var ctxAddAttribString = Loader.load("ctxAddAttribString","ossv");
   static var ctxAddAttribStringArray = Loader.load("ctxAddAttribStringArray","osov");
   static var ctxAddAttribTensor = Loader.load("ctxAddAttribTensor","osov");

}

