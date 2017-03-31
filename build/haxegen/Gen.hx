package haxegen;

class Gen
{
   public static function main()
   {
      cpp.Lib.pushDllSearchPath( "../ndll/" + cpp.Lib.getBinDirectory() );
      var genHxcppCode =cpp.Prime.load("tensorflow", "genHxcppCode", "sv", false);
      genHxcppCode("..");
   }

}

