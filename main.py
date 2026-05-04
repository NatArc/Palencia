
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify, lambdify, diff

def format_function(func_str):
    try:
        return str(sympify(func_str))
    except:
        return func_str


# ===================== plotter =====================
class Plotter:
    def __init__(self):
        self.x_sym = symbols('x')
        plt.style.use('grayscale')
        plt.rcParams['figure.facecolor'] = '#333333'
    
    def plot_function(self, func_str, x_min=-10, x_max=10):
        try:
            func_sym = sympify(func_str)
            func_np = lambdify(self.x_sym, func_sym, modules=['numpy'])
        
            x_vals = np.linspace(x_min, x_max, 1000)
            y_vals = func_np(x_vals)

            y_vals = np.nan_to_num(y_vals, nan=0.0)

            plt.figure(figsize=(12, 8))
            plt.plot(x_vals, y_vals, 'red', linewidth=3, label=f'Graph: {func_str}')
            plt.axhline(0, color='white', alpha=0.4)
            plt.axvline(0, color='white', alpha=0.4)
            plt.grid(True, alpha=0.3)
            plt.title(f'Graph of {func_str}', fontsize=16, pad=20, color='white')
            plt.xlabel('X-axis', fontsize=14, color='white')
            plt.ylabel('Y-values', fontsize=14, color='white')
            plt.legend()
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error plotting function: {e}")


# ===================== derivative =====================
class Derivative:
    def __init__(self):
        self.x_sym = symbols('x')
        plt.style.use('grayscale')
    
    def numerical_derivative(self, func_np, x_vals):
        h = x_vals[1] - x_vals[0]
        deriv = np.zeros_like(x_vals)
        deriv[1:-1] = (func_np(x_vals[2:]) - func_np(x_vals[:-2])) / (2 * h)
        deriv[0] = deriv[1]
        deriv[-1] = deriv[-2]
        return deriv
    
    def plot_with_derivative(self, func_str, x_min=-10, x_max=10):
        try:
            func_sym = sympify(func_str)
            deriv_sym = diff(func_sym, self.x_sym)

            pretty_f = str(func_sym)
            pretty_df = str(deriv_sym)

            func_np = lambdify(self.x_sym, func_sym, modules=['numpy'])
            
            x_vals = np.linspace(x_min, x_max, 1000)
            y_vals = func_np(x_vals)
            y_deriv = self.numerical_derivative(func_np, x_vals)

            y_vals = np.nan_to_num(y_vals, nan=0.0)
            y_deriv = np.nan_to_num(y_deriv, nan=0.0)
            
            area_deriv = np.trapezoid(np.abs(y_deriv), x_vals)

            plt.figure(figsize=(12, 8))

            plt.plot(x_vals, y_vals, 'red', linewidth=3, label='Original Function')
            plt.plot(x_vals, y_deriv, 'green', linewidth=3, label="Derivation")

            plt.axhline(0, color='white', alpha=0.4)
            plt.axvline(0, color='white', alpha=0.4)
            plt.grid(True, alpha=0.3)

            plt.title(
                f"Function | Derivation | Area ≈ {area_deriv:.3f}",
                fontsize=13, color='white'
            )

            plt.xlabel('X-axis', color='white')
            plt.ylabel('Values', color='white')

            plt.legend()
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error: {e}")


# ===================== integral =====================
class Integral:
    def __init__(self):
        self.x_sym = symbols('x')
        plt.style.use('grayscale')
    
    def numerical_integral(self, func_np, x_vals):
        y_vals = func_np(x_vals)
        y_vals = np.nan_to_num(y_vals, nan=0.0)

        integral = np.zeros_like(x_vals)
        
        for i in range(1, len(x_vals)):
            dx = x_vals[i] - x_vals[i-1]
            integral[i] = integral[i-1] + 0.5 * (y_vals[i] + y_vals[i-1]) * dx
            
        return integral
    
    def plot_with_integral(self, func_str, x_min=-10, x_max=10):
        try:
            func_sym = sympify(func_str)
            func_np = lambdify(self.x_sym, func_sym, modules=['numpy'])
            
            x_vals = np.linspace(x_min, x_max, 1000)
            y_vals = func_np(x_vals)
            y_integral = self.numerical_integral(func_np, x_vals)

            y_vals = np.nan_to_num(y_vals, nan=0.0)
            y_integral = np.nan_to_num(y_integral, nan=0.0)
            
            plt.figure(figsize=(12, 8))

            plt.plot(x_vals, y_vals, 'red', linewidth=3, label='Function')
            plt.plot(x_vals, y_integral, 'blue', linewidth=3, label='Integration')

            total_area = y_integral[-1] - y_integral[0]

            plt.fill_between(x_vals, 0, y_vals, alpha=0.2, color='red')

            plt.axhline(0, color='white', alpha=0.4)
            plt.axvline(0, color='white', alpha=0.4)
            plt.grid(True, alpha=0.3)

            pretty = format_function(func_str)
            plt.title(f"Function + Integration | Total = {total_area:.3f}", color='white')

            plt.xlabel('X-axis', color='white')
            plt.ylabel('Values', color='white')

            plt.legend()
            plt.tight_layout()
            plt.show()
            
            print(f"Total integration from {x_min} to {x_max}: {total_area:.4f}")

        except Exception as e:
            print(f"Error: {e}")


# ===================== visual =====================
class Visualization:
    def __init__(self):
        self.x_sym = symbols('x')
        plt.style.use('grayscale')
    
    def numerical_derivative(self, func_np, x_vals):
        h = x_vals[1] - x_vals[0]
        deriv = np.zeros_like(x_vals)
        deriv[1:-1] = (func_np(x_vals[2:]) - func_np(x_vals[:-2])) / (2 * h)
        deriv[0] = deriv[1]
        deriv[-1] = deriv[-2]
        return deriv
    
    def numerical_integral(self, func_np, x_vals):
        y_vals = func_np(x_vals)
        y_vals = np.nan_to_num(y_vals, nan=0.0)

        integral = np.zeros_like(x_vals)
        for i in range(1, len(x_vals)):
            dx = x_vals[i] - x_vals[i-1]
            integral[i] = integral[i-1] + 0.5 * (y_vals[i] + y_vals[i-1]) * dx
        
        integral = integral - integral[0]
        return integral
    
    def definite_integral_value(self, f_int, a, b, x_vals):
        idx_a = np.argmax(x_vals >= a)
        idx_b = np.argmax(x_vals >= b)
        return f_int[idx_b] - f_int[idx_a]
    
    def plot_all_with_areas(self, func_str, x_min=-10, x_max=10, a=None, b=None):
        try:
            func_sym = sympify(func_str)
            func_np = lambdify(self.x_sym, func_sym, modules=['numpy'])
            
            x_vals = np.linspace(x_min, x_max, 1000)
            f_x = func_np(x_vals)
            f_prime = self.numerical_derivative(func_np, x_vals)
            f_int = self.numerical_integral(func_np, x_vals)

            f_x = np.nan_to_num(f_x, nan=0.0)
            f_prime = np.nan_to_num(f_prime, nan=0.0)
            f_int = np.nan_to_num(f_int, nan=0.0)
            
            plt.figure(figsize=(14, 8))

            plt.plot(x_vals, f_x, 'red', linewidth=3, label='Function')
            plt.plot(x_vals, f_prime, 'green', linewidth=3, label="Differentiation")
            plt.plot(x_vals, f_int, 'blue', linewidth=3, label='Integration')

            title_text = "Function, Differentiation, Integration"

            if a is not None and b is not None:
                mask = (x_vals >= a) & (x_vals <= b)

                area_f = np.trapezoid(f_x[mask], x_vals[mask])
                area_p = np.trapezoid(np.abs(f_prime[mask]), x_vals[mask])
                area_i = self.definite_integral_value(f_int, a, b, x_vals)

                plt.fill_between(x_vals[mask], 0, f_x[mask], alpha=0.3, color='red')
                plt.fill_between(x_vals[mask], 0, np.abs(f_prime[mask]), alpha=0.3, color='green')
                plt.fill_between(x_vals[mask], 0, f_int[mask], alpha=0.3, color='blue')

                plt.axvline(a, color='white', linestyle='--', alpha=0.6)
                plt.axvline(b, color='white', linestyle='--', alpha=0.6)

                pretty = format_function(func_str)

                title_text = (
                    f"Function Area={area_f:.3f} | "
                    f"Differentiation Area={area_p:.3f} | "
                    f"Integration Area={area_i:.3f}"
                )

                print("\nArea Summary:")
                print(f" Interval [{a}, {b}]:")
                print(f"   Function:            {area_f:.4f}")
                print(f"   Differentiation:     {area_p:.4f}")
                print(f"   Integration:         {area_i:.4f}")

            plt.axhline(0, color='white', alpha=0.3)
            plt.axvline(0, color='white', alpha=0.3)
            plt.grid(True, alpha=0.3)

            plt.title(title_text, fontsize=13, color='white')
            plt.xlabel('X-axis', color='white')
            plt.ylabel('Y-values', color='white')

            plt.legend()
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Mode 4 Error: {e}")

# ===================== MAIN =====================
def get_x_range():
    while True:
        x_input = input("x-range : ").strip()
        if not x_input: return -10, 10
        try:
            parts = x_input.replace(',', ' ').split()
            if len(parts) == 1: return float(parts[0])-10, float(parts[0])+10
            elif len(parts) == 2: return float(parts[0]), float(parts[1])
        except:
            print("Examples: '-5 5', '0 10', '2'")
            continue

def get_area_range():
    while True:
        a_input = input("Area : ").strip()
        if not a_input: return None, None
        try:
            parts = a_input.replace(',', ' ').split()
            return float(parts[0]), float(parts[1])
        except:
            print("Examples: '0 2', '-1 3'")
            continue

def main():
    print("Calculus Graph")
    print("1: Function")
    print("2: Function + Derivative")
    print("3: Function + Integral")
    print("4: Function + Derivative + Integrals")
    print("5: Quit")
    
    plotter = Plotter()
    deriv = Derivative()
    integ = Integral()
    visual = Visualization()
    
    while True:
        choice = input("\nChoose (1/2/3/4/5): ").strip()
        
        if choice == '5':
            print("You Exited")
            break
            
        func = input("f(x): ").strip()
        if not func:
            continue
        
        x_min, x_max = get_x_range()
        print(f"x = [{x_min:.1f}, {x_max:.1f}]")
        
        if choice == '1':
            plotter.plot_function(func, x_min, x_max)
        elif choice == '2':
            deriv.plot_with_derivative(func, x_min, x_max)
        elif choice == '3':
            integ.plot_with_integral(func, x_min, x_max)
        elif choice == '4':
            a, b = get_area_range()
            visual.plot_all_with_areas(func, x_min, x_max, a, b)
        else:
            print("1,2,3,4 or 5")

if __name__ == "__main__":
    main()

