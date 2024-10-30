import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/settings_provider.dart';

class SettingsScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Settings'),
      ),
      body: Consumer<SettingsProvider>(
        builder: (context, settingsProvider, child) {
          return ListView(
            children: [
              SwitchListTile(
                title: const Text('Dark Mode'),
                value: settingsProvider.isDarkMode,
                onChanged: (value) {
                  settingsProvider.setDarkMode(value);
                },
              ),
              ListTile(
                title: const Text('Language'),
                subtitle: Text(settingsProvider.language),
                onTap: () {
                  // TODO: Implement language selection
                },
              ),
              ListTile(
                title: const Text('About'),
                onTap: () {
                  showAboutDialog(
                    context: context,
                    applicationName: 'ASL Translation App',
                    applicationVersion: '1.0.0',
                  );
                },
              ),
            ],
          );
        },
      ),
    );
  }
}