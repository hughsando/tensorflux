package haxegen;

using StringTools;

class Gen
{
   public static function main()
   {
      cpp.Lib.pushDllSearchPath( "../ndll/" + cpp.Lib.getBinDirectory() );
      var genCppCode =cpp.Prime.load("tensorflow", "genCppCode", "ssv", false);
      var dir = "gen/cppapi";
      sys.FileSystem.createDirectory(dir);
      genCppCode(dir+"/cppapi.cpp", dir+"/cppapi.h");

      var genHxcppCode =cpp.Prime.load("tensorflow", "genHxcppCode", "sssv", false);

      var path = "../modules/tensorflow/tensorflow/core/ops";
      var coreOps = sys.FileSystem.readDirectory(path);
      var registerOp = ~/REGISTER_OP\("(.*)"\)/;
      for(file in coreOps)
      {
         if (file.endsWith("_ops.cc"))
         {
            var ops = new Array<String>();
            var category = file.substr(0,file.length-"_ops.cc".length);
            if (category=="script")
               continue;
            for(line in sys.io.File.getContent(path+"/"+file).split("\n"))
            {
               if (registerOp.match(line))
               {
                  var op = registerOp.matched(1);
                  // Can't deal with output yet
                  if (op!="SparseSplit" && op!="ParseSingleSequenceExample" && op!="ParseExample")
                     ops.push(op);
               }
            }
            if (ops.length>0)
            {
               var filter = ":" + ops.join(":") + ":";
               var className = category.split("_").map( function(x)
                      return x.substr(0,1).toUpperCase() + x.substr(1)
                   ).join("") + "Ops";
               var classFile = '../tf/$className.hx';
               genHxcppCode(className, classFile, filter);
            }
         }
      }
      genHxcppCode("ConstOps", "../tf/ConstOps.hx", ":Const:");
   }

}

