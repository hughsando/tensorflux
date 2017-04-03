package tf;

typedef ConfigProto = {
   @:optional var cpuCount:Int;
   @:optional var gpuCount:Int;
}

abstract Session(Dynamic)
{
   public function new(?config:ConfigProto, ?target:String)
   {
      this = sesCreate(Context.currentHandle,config, target);
   }
   inline public function close()
   {
      sesClose(this);
      this = null;
   }
   public static function with(body:Session->Void)
   {
      var session = new Session();
      try
      {
         body(session);
      }
      catch(e:Dynamic)
      {
         session.close();
         throw e;
      }
      session.close();
   }

   public function runOutput(request:Output, ?feed_dict:Map<Output,Tensor> /* TODO:opts,meta */) : Tensor
   {
      return run([request],feed_dict)[0];
   }
   public function runOutputs(fetches:Array<Output>, ?feedOutputs:Array<Output>, ?feedValues:Array<Tensor> /* TODO:opts,meta */) : Array<Tensor>
   {
      var oCount = feedOutputs==null ? -1 : feedOutputs.length;
      var vCount = feedValues==null ? -1 : feedValues.length;
      if (oCount!=vCount)
         throw "mismatched feed name and value counts";
      var result = new Array<Tensor>();
      var n = fetches.length;
      if (n==0)
         return [];
      result[n-1] = null;
      sesRun(this, fetches, feedOutputs, feedValues, result);
      return result;
   }

   public function run(fetches:Array<Output>, ?feed_dict:Map<Output,Tensor> /* TODO:opts,meta */) : Array<Tensor>
   {
      if (feed_dict==null)
         return runOutputs(fetches);
      var inputs = new Array<Output>();
      var values = new Array<Tensor>();
      for(key in feed_dict.keys())
      {
         inputs.push(key);
         values.push( feed_dict.get(key) );
      }
      return runOutputs(fetches, inputs, values);
   }

   static var sesCreate = Loader.load("sesCreate","ooso");
   static var sesRun = Loader.load("sesRun","ooooov");
   static var sesClose = Loader.load("sesClose","ov");
}
