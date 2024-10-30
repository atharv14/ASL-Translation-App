import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/translation_provider.dart';

class TranslationScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('ASL Translation'),
      ),
      body: Consumer<TranslationProvider>(
        builder: (context, translationProvider, child) {
          final translation = translationProvider.currentTranslation;
          if (translation == null) {
            return const Center(child: Text('No translation available'));
          }
          return Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('ASL Input:', style: TextStyle(fontWeight: FontWeight.bold)),
                Text(translation.aslInput),
                const SizedBox(height: 20),
                const Text('English Translation:', style: TextStyle(fontWeight: FontWeight.bold)),
                Text(translation.englishOutput),
                const SizedBox(height: 20),
                Text('Timestamp: ${translation.timestamp.toString()}'),
              ],
            ),
          );
        },
      ),
    );
  }
}