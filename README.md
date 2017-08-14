# tensorflux
Haxe build tool for tensor flow

Build
-----
`cd build && haxe --run Build.hx all`

or

`cd build && haxe --run Build.hx all -DGPU`

This will create the dll, and run a test program to see if it works.

Why a script?
-------------
The build procedure is complicated - you need to build a tool(protoc) to build code to build a tool(proto_text) to build code to build the dll.  A Haxe program (haxegen.Gen) then uses the dll (plus haxe generated code, of course!) to generate the "*Ops.hx" files, which then completes the host building. The tooling is done once on the host, and the dll may be build again for different client.
