// import 'package:flutter/material.dart';
// import 'package:flutter_test/flutter_test.dart';
// import 'package:provider/provider.dart';
// import 'package:frontend/main.dart';
// import 'package:frontend/providers/auth_providers.dart';
// import 'package:mockito/mockito.dart';
// import 'package:firebase_auth/firebase_auth.dart';
//
// class MockAuthProvider extends Mock implements AuthProvider {}
// class MockUser extends Mock implements User {}
//
// void main() {
//   group('ASL Translation App Tests', () {
//     late MockAuthProvider mockAuthProvider;
//
//     setUp(() {
//       mockAuthProvider = MockAuthProvider();
//     });
//
//     testWidgets('Login screen is shown when user is not authenticated', (WidgetTester tester) async {
//       when(mockAuthProvider.user).thenReturn(null);
//
//       await tester.pumpWidget(
//         MultiProvider(
//           providers: [
//             ChangeNotifierProvider<AuthProvider>.value(value: mockAuthProvider),
//           ],
//           child: ASLTranslationApp(),
//         ),
//       );
//
//       expect(find.text('Login'), findsOneWidget);
//       expect(find.text('Home'), findsNothing);
//     });
//
//     testWidgets('Home screen is shown when user is authenticated', (WidgetTester tester) async {
//       final mockUser = MockUser();
//       when(mockAuthProvider.user).thenReturn(mockUser);
//
//       await tester.pumpWidget(
//         MultiProvider(
//           providers: [
//             ChangeNotifierProvider<AuthProvider>.value(value: mockAuthProvider),
//           ],
//           child: ASLTranslationApp(),
//         ),
//       );
//
//       expect(find.text('Home'), findsOneWidget);
//       expect(find.text('Login'), findsNothing);
//     });
//   });
// }