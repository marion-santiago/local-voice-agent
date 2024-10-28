import React, { ButtonHTMLAttributes, ReactNode } from "react";

export type ButtonProps = {
  children: ReactNode;
  className?: string;
  disabled?: boolean;
  state?: "primary" | "secondary" | "destructive";
  size?: "small" | "medium" | "large";
} & ButtonHTMLAttributes<HTMLButtonElement>;

export const Button: React.FC<ButtonProps> = ({
  children,
  className,
  disabled,
  state = "primary",
  size = "small",
  ...allProps
}) => {
  let buttonStyles =
    "hover:shadow-solid-offset-hover active:shadow-solid-offset-active shadow-solid-offset border-black";
  if (state === "primary") {
    buttonStyles += ` bg-cartesia-500 hover:bg-cartesia-600 text-white`;
  } else if (state === "secondary") {
    buttonStyles += ` bg-transparent text-black hover:bg-gray-100`;
  } else if (state === "destructive") {
    buttonStyles =
      "hover:shadow-solid-offset-destructive-hover active:shadow-solid-offset-destructive-active shadow-solid-offset-destructive border-red-500 bg-red-50 hover:bg-red-100 text-red-600";
  }

  let sizeStyles = "text-xs px-2 py-[6px] font-semibold ";
  if (size === "large") {
    sizeStyles = "text-lg px-6 py-4 font-regular tracking-wider";
  } else if (size === "medium") {
    sizeStyles = "text-sm px-2 py-2 font-semibold";
  }

  buttonStyles += " " + sizeStyles;

  return (
    <button
      className={`hover:-translate-y-[1px] hover:-translate-x-[1px] active:translate-y-[2px] active:translate-x-[2px] flex flex-row ${
        disabled ? "pointer-events-none" : ""
      } ${size} font-mono uppercase ${buttonStyles} transition-all border-2 ease-out duration-250 ${className}`}
      {...allProps}
    >
      {children}
    </button>
  );
};
