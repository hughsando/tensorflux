# tensorflux
Haxe build tool for tensor flow

Build
-----
`cd build && haxe --run Build.hx all`

This will create the dll, which does not do anything useful yet.

Why a script?
-------------
The build procedure is complicated - you need to build a tool(protoc) to build code to build a tool(proto_text) to build code to build the dll.  The tooling is done once on the host, and the dll is built for the client.
