import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import Button from '../Button';

describe('Button Component', () => {
  test('renders button with text', () => {
    render(<Button>Click Me</Button>);
    expect(screen.getByText('Click Me')).toBeInTheDocument();
  });

  test('applies primary variant styles by default', () => {
    render(<Button>Primary Button</Button>);
    const button = screen.getByText('Primary Button');
    expect(button).toHaveClass('bg-blue-600');
    expect(button).toHaveClass('text-white');
  });

  test('applies secondary variant styles when specified', () => {
    render(<Button variant="secondary">Secondary Button</Button>);
    const button = screen.getByText('Secondary Button');
    expect(button).toHaveClass('bg-gray-200');
    expect(button).toHaveClass('text-gray-800');
  });

  test('calls onClick handler when clicked', () => {
    const handleClick = jest.fn();
    render(<Button onClick={handleClick}>Clickable</Button>);
    fireEvent.click(screen.getByText('Clickable'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  test('displays loading spinner when isLoading is true', () => {
    render(<Button isLoading>Loading Button</Button>);
    expect(screen.getByText('Loading Button')).toBeInTheDocument();
    expect(screen.getByRole('button')).toBeDisabled();
    expect(document.querySelector('svg.animate-spin')).toBeInTheDocument();
  });

  test('renders with left icon when provided', () => {
    render(
      <Button leftIcon={<span data-testid="left-icon">🔍</span>}>
        With Left Icon
      </Button>
    );
    expect(screen.getByTestId('left-icon')).toBeInTheDocument();
  });

  test('renders with full width when fullWidth is true', () => {
    render(<Button fullWidth>Full Width Button</Button>);
    expect(screen.getByText('Full Width Button')).toHaveClass('w-full');
  });
});