import 'package:flutter/foundation.dart';
import '../models/translation.dart';

class TranslationProvider with ChangeNotifier {
  Translation? _currentTranslation;

  Translation? get currentTranslation => _currentTranslation;

  void setTranslation(Translation translation) {
    _currentTranslation = translation;
    notifyListeners();
  }

  Future<void> performTranslation(String aslInput) async {
    // TODO: Implement API call to backend for translation
    // For now, we'll use a mock translation
    _currentTranslation = Translation(
      id: DateTime.now().millisecondsSinceEpoch,
      aslInput: aslInput,
      englishOutput: 'Mock translation for: $aslInput',
      timestamp: DateTime.now(),
    );
    notifyListeners();
  }
}