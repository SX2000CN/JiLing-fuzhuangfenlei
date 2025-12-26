# theme-system Spec Delta

## MODIFIED Requirements

### Requirement: Light Theme

The system SHALL provide a light theme with WCAG 2.1 AA compliant contrast ratios.

All text colors SHALL have a minimum contrast ratio of 4.5:1 against their background.
All icon colors SHALL have a minimum contrast ratio of 3:1 against their background.
Selected item backgrounds SHALL provide sufficient contrast for white text (TEXT_INVERSE).

#### Scenario: Light theme text contrast
- **WHEN** light theme is active
- **THEN** `text.primary` (#1F1F1F) on `bg.primary` (#FFFFFF) has contrast ratio ≥ 14:1
- **AND** `text.secondary` (#505050) on `bg.primary` (#FFFFFF) has contrast ratio ≥ 7:1
- **AND** `text.muted` (#6B6B6B) on `bg.primary` (#FFFFFF) has contrast ratio ≥ 4.5:1
- **AND** `disabled.text` (#767676) on `disabled.bg` (#F5F5F5) has contrast ratio ≥ 4.5:1

#### Scenario: Light theme icon contrast
- **WHEN** light theme is active
- **THEN** `icon.default` (#3B3B3B) on any background token has contrast ratio ≥ 3:1

#### Scenario: Light theme selected item readability
- **WHEN** light theme is active
- **AND** an item is selected (e.g., table row)
- **THEN** `selected.bg` (#0060C0) provides contrast ratio ≥ 4.5:1 for `text.inverse` (#FFFFFF)

#### Scenario: Light theme log readability
- **WHEN** light theme is active
- **THEN** `log.warning` (#956700) on `bg.primary` (#FFFFFF) has contrast ratio ≥ 4.5:1
- **AND** `log.debug` (#006600) on `bg.primary` (#FFFFFF) has contrast ratio ≥ 4.5:1
