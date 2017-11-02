package tf;

class Operation
{
   var handle:Dynamic;

   public var name(get,null):String;
   public var type(get,null):String;
   public var device(get,null):String;

   public var inputCount(get,null):Int;
   public var outputCount(get,null):Int;

   public function new(inHandle:Dynamic)
   {
      handle = inHandle;
   }

   public function getInputType(index:Int) : Int
   {
      return opInputType(handle,index);
   }

   public function getOutput(inIndex:Int) : Output
   {
      return opOutput(handle,inIndex);
   }

   public function getInput(inIndex:Int) : Output
   {
      return opInput(handle,inIndex);
   }

   public function toString() return name;

   function get_name() : String return opGetName(handle);
   function get_type() : String return opGetType(handle);
   function get_device() : String return opGetDevice(handle);

   function get_inputCount() : Int return opInputCount(handle);

   function get_outputCount() : Int return opOutputCount(handle);


   static var opGetName = Loader.load("opGetName","os");
   static var opGetType = Loader.load("opGetType","os");
   static var opGetDevice = Loader.load("opGetDevice","os");

   static var opInputCount = Loader.load("opInputCount","oi");
   static var opInputType = Loader.load("opInputType","oii");
   static var opInput = Loader.load("opInput","oio");

   static var opOutputCount = Loader.load("opOutputCount","oi");
   static var opOutput = Loader.load("opOutput","oio");
}

