module.exports = {
  root: true,
  env: {browser: true, es2020: true},
  extends: [
    'eslint:recommended',
    'google',
    'plugin:@typescript-eslint/recommended',
    'plugin:react-hooks/recommended',
  ],
  ignorePatterns: ['dist', '.eslintrc.cjs'],
  parser: '@typescript-eslint/parser',
  plugins: ['react-refresh'],
  rules: {
    'react-refresh/only-export-components': [
      'warn',
      {allowConstantExport: true},
    ],
    '@typescript-eslint/no-unused-vars': 'warn',
    "require-jsdoc": "off",
    "max-len": "off",
    "no-unused-vars": "warn",
    "spaced-comment": "warn",
    "valid-jsdoc": "warn",
    "react-hooks/rules-of-hooks": "warn", // Checks rules for Hooks
    "react-hooks/exhaustive-deps": "warn", // Checks effect dependencies
    "no-invalid-this": "warn",
    "camelcase": "off",
    "new-cap": "off",
    "@typescript-eslint/ban-ts-comment": "off",
    "@typescript-eslint/no-explicit-any": "off",
    "react-hooks/rules-of-hooks": "off",
    "valid-jsdoc": "off",
    // "@typescript-eslint/no-unused-vars":
  },
}
