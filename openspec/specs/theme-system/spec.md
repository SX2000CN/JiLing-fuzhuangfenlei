# theme-system Specification

## Purpose
TBD - created by archiving change refactor-modular-ui. Update Purpose after archive.
## Requirements
### Requirement: Theme Manager

The system SHALL provide a ThemeManager singleton that manages application themes.

The ThemeManager SHALL:
- Support at least two themes: "dark" and "light"
- Provide a method to get current theme colors by token
- Emit a signal when theme changes
- Persist theme preference

#### Scenario: Get theme color
- **WHEN** `ThemeManager.get_color("bg.primary")` is called
- **THEN** the correct color value for the current theme is returned

#### Scenario: Switch theme
- **WHEN** `ThemeManager.set_theme("light")` is called
- **THEN** the theme_changed signal is emitted with "light"
- **AND** subsequent `get_color()` calls return light theme values

### Requirement: Color Token System

The system SHALL use semantic color tokens instead of hardcoded color values.

Tokens SHALL include:
- Background tokens: `bg.primary`, `bg.secondary`, `bg.sidebar`, `bg.input`
- Text tokens: `text.primary`, `text.secondary`, `text.muted`
- Accent tokens: `accent`, `accent.hover`
- Border tokens: `border`, `border.focus`
- State tokens: `hover.bg`, `selected.bg`

#### Scenario: Token consistency
- **WHEN** a component needs a background color
- **THEN** it uses a token (e.g., `bg.primary`) instead of a hardcoded hex value

#### Scenario: Theme-aware styling
- **WHEN** the theme changes from dark to light
- **THEN** all components using tokens automatically reflect the new colors

### Requirement: Dark Theme

The system SHALL provide a dark theme with VS Code-inspired colors.

#### Scenario: Dark theme colors
- **WHEN** dark theme is active
- **THEN** `bg.primary` returns "#1E1E1E"
- **AND** `text.primary` returns "#CCCCCC"
- **AND** `accent` returns "#007FD4"

### Requirement: Light Theme

The system SHALL provide a light theme with appropriate contrast.

#### Scenario: Light theme colors
- **WHEN** light theme is active
- **THEN** `bg.primary` returns "#FFFFFF"
- **AND** `text.primary` returns "#333333"
- **AND** `accent` returns "#0066B8"

#### Scenario: Light theme readability
- **WHEN** light theme is active
- **THEN** text has sufficient contrast against backgrounds (WCAG AA compliant)

### Requirement: Dynamic Style Generation

StyleSheets SHALL be generated dynamically based on current theme tokens.

#### Scenario: Style regeneration on theme change
- **WHEN** theme changes
- **THEN** component stylesheets are regenerated with new token values
- **AND** visible components update their appearance

### Requirement: Settings Row Highlight Fix

The SettingsRow component SHALL display hover highlight with proper padding between highlight edge and text.

#### Scenario: Hover highlight padding
- **WHEN** mouse hovers over a settings row
- **THEN** highlight background is visible
- **AND** there is at least 12px padding between highlight edge and text on all sides

#### Scenario: Text alignment with section title
- **WHEN** settings page is displayed
- **THEN** settings row text aligns with section title text (both at 24px from left edge)

