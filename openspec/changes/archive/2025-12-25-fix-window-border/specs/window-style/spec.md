## MODIFIED Requirements

### Requirement: 窗口边框显示

窗口 SHALL 在非最大化状态下显示完整的边框，边框不被任何子控件遮挡。

#### Scenario: 正常状态显示边框
- **WHEN** 窗口处于正常（非最大化）状态
- **THEN** 窗口四周显示完整的圆角边框
- **AND** 边框在所有子控件之上绘制，不被遮挡

#### Scenario: 最大化状态隐藏边框
- **WHEN** 窗口处于最大化状态
- **THEN** 窗口不显示边框
- **AND** 窗口填满屏幕工作区域

#### Scenario: 边框颜色可配置
- **WHEN** 修改 `BORDER_COLOR` 常量
- **THEN** 边框颜色相应改变
