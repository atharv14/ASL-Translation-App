import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/translation_provider.dart';

// class HomeScreen extends StatelessWidget {
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(
//         title: const Text('ASL Translation App'),
//       ),
//       body: Center(
//         child: Column(
//           mainAxisAlignment: MainAxisAlignment.center,
//           children: <Widget>[
//             ElevatedButton(
//               child: const Text('Start Translation'),
//               onPressed: () {
//                 Navigator.pushNamed(context, '/camera');
//               },
//             ),
//             const SizedBox(height: 20),
//             ElevatedButton(
//               child: const Text('Settings'),
//               onPressed: () {
//                 Navigator.pushNamed(context, '/settings');
//               },
//             ),
//           ],
//         ),
//       ),
//     );
//   }
// }

import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'camera_screen.dart';

class HomeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('ASL Translation App'),
      ),
      body: Center(
        child: ElevatedButton(
          child: Text('Start Translation'),
          onPressed: () async {
            if (await Permission.camera.request().isGranted) {
              Navigator.of(context).push(
                MaterialPageRoute(builder: (_) => CameraScreen()),
              );
            } else {
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(content: Text('Camera permission is required')),
              );
            }
          },
        ),
      ),
    );
  }
}