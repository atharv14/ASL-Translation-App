import 'package:flutter/material.dart';
import 'package:logger/logger.dart';

class ErrorHandler {
  static final Logger _logger = Logger();

  static void handleError(BuildContext context, dynamic error, StackTrace? stackTrace) {
    _logger.e('An error occurred', error: error, stackTrace: stackTrace);

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('An error occurred. Please try again.'),
        backgroundColor: Colors.red,
      ),
    );
  }
}