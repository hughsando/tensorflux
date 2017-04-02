package tf;

class Scope
{
   var path:String;
   var vars = new Map<String,Bool>();

   public function new(inPath:String)
   {
      path = inPath;
   }
   public function addUnique(name:String)
   {
      if (vars.exists(name))
      {
         var idx = 1;
         while(true)
         {
            var test = name + idx;
            if (!vars.exists(test))
            {
               name = test;
               break;
            }
            idx++;
         }
      }
      vars.set(name,true);

      return path + name;
   }
}
