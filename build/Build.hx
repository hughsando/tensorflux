import sys.FileSystem;

class Build
{
   static var haxelibExtra:Array<String> = [ ];
   static var builds = ["libprotobuf", "protoc", "libpbcc", "proto_text", "libpb_text",
         "tensorflow", "haxegen", "test" ];
   static var toolExt = Sys.systemName()=="Windows" ? ".exe" : "";
   static var commandError = false;

   public static function command(exe:String, args:Array<String>)
   {
      if (exe=="haxelib")
         args = args.concat(haxelibExtra);

      Sys.println(exe +" " + args.join(' '));
      if (Sys.command(exe,args)!=0)
         commandError = true;
   }

   public static function libprotobufBuild()
   {
      sys.FileSystem.createDirectory("lib");
      command("haxelib", ["run","hxcpp","libprotobuf.xml","-Dstatic_link" ] );
   }

   public static function protocBuild()
   {
      sys.FileSystem.createDirectory("bin");
      command("haxelib", ["run","hxcpp","protoc.xml" ] );
   }

   public static function haxegenBuild()
   {
      command("haxe", ["-main","haxegen.Gen", "-cpp", "obj", "-D", "HXCPP_M64" ] );
      if (commandError)
         return;
      command('obj/Gen$toolExt', []);
   }

   public static function libpbccBuild()
   {
      var pbCc = [ 
         "tensorflow/core/util/test_log.pb.cc",
         "tensorflow/core/util/saved_tensor_slice.pb.cc",
         "tensorflow/core/util/event.pb.cc",
         "tensorflow/core/util/memmapped_file_system.pb.cc",
         "tensorflow/core/protobuf/tensorflow_server.pb.cc",
         "tensorflow/core/protobuf/saver.pb.cc",
         "tensorflow/core/protobuf/queue_runner.pb.cc",
         "tensorflow/core/protobuf/named_tensor.pb.cc",
         "tensorflow/core/protobuf/meta_graph.pb.cc",
         "tensorflow/core/protobuf/config.pb.cc",
         "tensorflow/core/protobuf/debug.pb.cc",
         "tensorflow/core/protobuf/tensor_bundle.pb.cc",
         "tensorflow/core/lib/core/error_codes.pb.cc",
         "tensorflow/core/framework/versions.pb.cc",
         "tensorflow/core/framework/variable.pb.cc",
         "tensorflow/core/framework/types.pb.cc",
         "tensorflow/core/framework/tensor_slice.pb.cc",
         "tensorflow/core/framework/tensor_shape.pb.cc",
         "tensorflow/core/framework/tensor_description.pb.cc",
         "tensorflow/core/framework/tensor.pb.cc",
         "tensorflow/core/framework/summary.pb.cc",
         "tensorflow/core/framework/step_stats.pb.cc",
         "tensorflow/core/framework/resource_handle.pb.cc",
         "tensorflow/core/framework/remote_fused_graph_execute_info.pb.cc",
         "tensorflow/core/framework/reader_base.pb.cc",
         "tensorflow/core/framework/op_def.pb.cc",
         "tensorflow/core/framework/node_def.pb.cc",
         "tensorflow/core/framework/log_memory.pb.cc",
         "tensorflow/core/framework/kernel_def.pb.cc",
         "tensorflow/core/framework/graph_transfer_info.pb.cc",
         "tensorflow/core/framework/graph.pb.cc",
         "tensorflow/core/framework/function.pb.cc",
         "tensorflow/core/framework/device_attributes.pb.cc",
         "tensorflow/core/framework/cost_graph.pb.cc",
         "tensorflow/core/framework/attr_value.pb.cc",
         "tensorflow/core/framework/allocation_description.pb.cc",
         "tensorflow/core/example/feature.pb.cc",
         "tensorflow/core/example/example.pb.cc",
         "tensorflow/core/protobuf/rewriter_config.pb.cc",
         "tensorflow/core/framework/op_gen_overrides.pb.cc",
         "tensorflow/core/protobuf/saved_model.pb.cc",
      ];
      FileSystem.createDirectory("gen");

      var lines = ["<xml><files id='gen-files' dir='gen' >"];
      lines.push('<compilerflag value="-DHAVE_PTHREAD" unless="windows" />');
      lines.push('<compilerflag value="-Igen" />');
      lines.push('<compilerflag value="-I../modules/protobuf/src" />');
      for(file in pbCc)
      {
         lines.push( '   <file name="$file" />');
         if (!FileSystem.exists('gen/$file'))
         {
            var proto = file.substr(0,file.length-"pb.cc".length) + "proto";
            command('bin/protoc$toolExt', ["-I../modules/tensorflow", "-I../modules/protobuf/src", '../modules/tensorflow/$proto', "--cpp_out","gen" ]);
         }
      }
      lines.push("</files></xml>");

      sys.io.File.saveContent("gen/files.xml", lines.join("\n"));


      command("haxelib", ["run","hxcpp","libpbcc.xml","-Dstatic_link"] );
   }


  /*
  proto_text is a tool that converts protobufs into a form we can use more
  compactly within TensorFlow. It's a bit like protoc, but is designed to
  produce a much more minimal result so we can save binary space.
  We have to build it on the host system first so that we can create files
  that are needed for the runtime building.
  */


   public static function proto_textBuild()
   {
      sys.FileSystem.createDirectory("bin");
      command("haxelib", ["run","hxcpp","proto_text.xml" ] );
   }

   public static function libpb_textBuild()
   {
      var pb_text = [ 
         "tensorflow/core/util/saved_tensor_slice.pb_text.cc",
         "tensorflow/core/util/memmapped_file_system.pb_text.cc",
         "tensorflow/core/protobuf/saver.pb_text.cc",
         "tensorflow/core/protobuf/config.pb_text.cc",
         "tensorflow/core/protobuf/debug.pb_text.cc",
         "tensorflow/core/protobuf/rewriter_config.pb_text.cc",
         "tensorflow/core/protobuf/tensor_bundle.pb_text.cc",
         "tensorflow/core/lib/core/error_codes.pb_text.cc",
         "tensorflow/core/framework/versions.pb_text.cc",
         "tensorflow/core/framework/types.pb_text.cc",
         "tensorflow/core/framework/tensor_slice.pb_text.cc",
         "tensorflow/core/framework/tensor_shape.pb_text.cc",
         "tensorflow/core/framework/tensor_description.pb_text.cc",
         "tensorflow/core/framework/tensor.pb_text.cc",
         "tensorflow/core/framework/summary.pb_text.cc",
         "tensorflow/core/framework/step_stats.pb_text.cc",
         "tensorflow/core/framework/resource_handle.pb_text.cc",
         "tensorflow/core/framework/remote_fused_graph_execute_info.pb_text.cc",
         "tensorflow/core/framework/op_def.pb_text.cc",
         "tensorflow/core/framework/node_def.pb_text.cc",
         "tensorflow/core/framework/log_memory.pb_text.cc",
         "tensorflow/core/framework/kernel_def.pb_text.cc",
         "tensorflow/core/framework/graph_transfer_info.pb_text.cc",
         "tensorflow/core/framework/graph.pb_text.cc",
         "tensorflow/core/framework/function.pb_text.cc",
         "tensorflow/core/framework/device_attributes.pb_text.cc",
         "tensorflow/core/framework/cost_graph.pb_text.cc",
         "tensorflow/core/framework/attr_value.pb_text.cc",
         "tensorflow/core/framework/allocation_description.pb_text.cc",
         "tensorflow/core/example/feature.pb_text.cc",
         "tensorflow/core/example/example.pb_text.cc",
      ];

      FileSystem.createDirectory("gen/tensorflow/core");

      // proto_text is very picky about which directory it is run in...
      var here = Sys.getCwd();
      Sys.setCwd("../modules/tensorflow");

      var lines = ["<xml><files id='pb-text-gen-files' dir='gen' >"];
      lines.push('<compilerflag value="-DHAVE_PTHREAD" unless="windows" />');
      lines.push('<compilerflag value="-Igen" />');
      lines.push('<compilerflag value="-I../modules/protobuf/src" />');
      lines.push('<compilerflag value="-I../modules/tensorflow" />');
      lines.push('<compilerflag value="-DCOMPILER_MSVC" if="isMsvc" />');
      lines.push('<compilerflag value="-DPLATFORM_WINDOWS" if="windows" />');
      for(file in pb_text)
      {
         lines.push( '   <file name="$file" />');
         if (!FileSystem.exists('../../build/gen/$file'))
         {
            var proto = file.substr(0,file.length-"pb_text.cc".length) + "proto";
            command('../../build/bin/proto_text$toolExt', ["../../build/gen/tensorflow/core", "tensorflow/core", "tensorflow/tools/proto_text/placeholder.txt", proto ]);
         }
      }
      lines.push("</files></xml>");

      Sys.setCwd(here);

      sys.io.File.saveContent("gen/proto_text_files.xml", lines.join("\n"));

      command("haxelib", ["run","hxcpp","libpb_text.xml","-Dstatic_link"] );
   }

   public static function tensorflowBuild()
   {
      command("haxelib", ["run","hxcpp","tensorflow.xml" ] );
   }

   public static function testBuild()
   {
      var here = Sys.getCwd();
      Sys.setCwd("../test/smoke");

      command("haxe", ["compile.hxml" ] );
      if (!commandError)
         command('cpp/Test$toolExt', [] );
      Sys.setCwd(here);
   }


   static function deleteDirRecurse(dir:String)
   {
      for(file in FileSystem.readDirectory(dir))
      {
         var path = dir + "/" + file;
         if (FileSystem.isDirectory(path))
            deleteDirRecurse(path);
         else
            FileSystem.deleteFile(path);
      }
      FileSystem.deleteDirectory(dir);
   }


   public static function cleanBuild()
   {
      for(dir in ["gen", "obj", "lib", "bin" ])
      {
         try {
            deleteDirRecurse(dir);
         } catch(d:Dynamic) { }
      }
   }

   public static function main()
   {
      haxelibExtra.push("-DHXCPP_M64");

      var option = Sys.args()[0];
      if (option=="all")
      {
         for(build in builds)
         {
            Sys.println('$build...');
            Reflect.field(Build, build + "Build")();
            if (commandError)
            {
               Sys.println('There were errors building $build');
               Sys.exit(-1);
            }
         }
      }
      else if (option=="clean")
      {
         cleanBuild();
      }
      else
      {
         if (builds.indexOf(option)<0)
         {
            Sys.println("Usage: Build target");
            Sys.println(' target = one of :$builds or "all" or "clean"');
            Sys.exit(-1);
         }

         Reflect.field(Build, option + "Build")();
      }
   }
}
