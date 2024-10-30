// import 'package:flutter/material.dart';
// import 'package:provider/provider.dart';
// import 'package:firebase_core/firebase_core.dart';
// import 'screens/home_screen.dart';
// import 'screens/login_screen.dart';
// import 'providers/auth_providers.dart';
// import 'providers/translation_provider.dart';
// import 'providers/settings_provider.dart';

// void main() async {
//   WidgetsFlutterBinding.ensureInitialized();
//   await Firebase.initializeApp();
//   runApp(
//     MultiProvider(
//       providers: [
//         ChangeNotifierProvider(create: (_) => AuthProvider()),
//         ChangeNotifierProvider(create: (_) => TranslationProvider()),
//         ChangeNotifierProvider(create: (_) => SettingsProvider()),
//       ],
//       child: ASLTranslationApp(),
//     ),
//   );
// }
//
// class ASLTranslationApp extends StatelessWidget {
//   @override
//   Widget build(BuildContext context) {
//     return Consumer<AuthProvider>(
//       builder: (context, authProvider, _) {
//         return MaterialApp(
//           title: 'ASL Translation App',
//           theme: ThemeData(
//             primarySwatch: Colors.blue,
//             visualDensity: VisualDensity.adaptivePlatformDensity,
//           ),
//           home: authProvider.user == null ? LoginScreen() : HomeScreen(),
//           routes: {
//             '/home': (context) => HomeScreen(),
//             '/login': (context) => LoginScreen(),
//             // Add other routes here
//           },
//         );
//       },
//     );
//   }
// }

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'screens/home_screen.dart';
import 'providers/translation_provider.dart';

void main() {
  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => TranslationProvider()),
      ],
      child: ASLTranslationApp(),
    ),
  );
}

class ASLTranslationApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ASL Translation App',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: HomeScreen(),
    );
  }
}